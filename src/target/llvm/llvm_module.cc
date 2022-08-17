/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file llvm_module.cc
 * \brief LLVM runtime module for TVM
 */
#ifdef TVM_LLVM_VERSION

#include "llvm_module.h"

#include <dmlc/io.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>  // Force linking of MCJIT
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <tvm/ir/module.h>
#include <tvm/relay/runtime.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/metadata.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/support/with.h>
#include <tvm/target/codegen.h>
#include <tvm/target/target.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "../../runtime/file_utils.h"
#include "../../runtime/library_module.h"
#include "../func_registry_generator.h"
#include "codegen_blob.h"
#include "codegen_cpu.h"
#include "codegen_llvm.h"
#include "llvm_instance.h"

namespace tvm {
namespace codegen {

using runtime::PackedFunc;
using runtime::TVMArgs;
using runtime::TVMRetValue;

class LLVMModuleNode final : public runtime::ModuleNode {
 public:
  ~LLVMModuleNode();

  const char* type_key() const final { return "llvm"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;

  void SaveToFile(const std::string& file_name, const std::string& format) final;
  void SaveToBinary(dmlc::Stream* stream) final;
  std::string GetSource(const std::string& format) final;

  void Init(const IRModule& mod, const Target& target);
  void Init(std::unique_ptr<llvm::Module> module, std::unique_ptr<LLVMInstance> llvm_instance);
  void LoadIR(const std::string& file_name);
  bool IsDSOExportable() const final { return true; }

  bool ImplementsFunction(const String& name, bool query_imports) final;

 private:
  void LazyInitJIT();
  bool IsCompatibleWithHost(const llvm::TargetMachine* tm) const;
  void* GetGlobalAddr(const std::string& name, const LLVMTarget& llvm_target) const;
  void* GetFunctionAddr(const std::string& name, const LLVMTarget& llvm_target) const;

  // The LLVM scope object.
  std::unique_ptr<LLVMInstance> llvm_instance_;
  // JIT lock
  std::mutex mutex_;
  // execution engine
  llvm::ExecutionEngine* ee_{nullptr};
  // The raw pointer to the module.
  llvm::Module* module_{nullptr};
  // The unique_ptr owning the module. This becomes empty once JIT has been initialized
  // (EngineBuilder takes ownership of the module).
  std::unique_ptr<llvm::Module> module_owning_ptr_;
  /* \brief names of the functions declared in this module */
  Array<String> function_names_;
};

LLVMModuleNode::~LLVMModuleNode() {
  if (ee_ != nullptr) {
    ee_->runStaticConstructorsDestructors(true);
    delete ee_;
  }
  module_owning_ptr_.reset();
}

PackedFunc LLVMModuleNode::GetFunction(const std::string& name,
                                       const ObjectPtr<Object>& sptr_to_self) {
  if (name == "__tvm_is_system_module") {
    bool flag = (module_->getFunction("__tvm_module_startup") != nullptr);
    return PackedFunc([flag](TVMArgs args, TVMRetValue* rv) { *rv = flag; });
  } else if (name == "get_func_names") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->function_names_; });
  } else if (name == "get_symbol") {
    return PackedFunc(nullptr);
  } else if (name == "get_const_vars") {
    return PackedFunc(nullptr);
  } else if (name == "_get_target_string") {
    std::string target_string = LLVMTarget::GetTargetMetadata(*module_);
    return PackedFunc([target_string](TVMArgs args, TVMRetValue* rv) { *rv = target_string; });
  }
  if (ee_ == nullptr) LazyInitJIT();

  std::lock_guard<std::mutex> lock(mutex_);

  TVMBackendPackedCFunc faddr;
  With<LLVMTarget> llvm_target(*llvm_instance_, LLVMTarget::GetTargetMetadata(*module_));
  if (name == runtime::symbol::tvm_module_main) {
    const char* entry_name = reinterpret_cast<const char*>(
        GetGlobalAddr(runtime::symbol::tvm_module_main, *llvm_target));
    ICHECK(entry_name != nullptr) << "Symbol " << runtime::symbol::tvm_module_main
                                  << " is not presented";
    faddr = reinterpret_cast<TVMBackendPackedCFunc>(GetFunctionAddr(entry_name, *llvm_target));
  } else {
    faddr = reinterpret_cast<TVMBackendPackedCFunc>(GetFunctionAddr(name, *llvm_target));
  }
  if (faddr == nullptr) return PackedFunc();
  return WrapPackedFunc(faddr, sptr_to_self);
}

void LLVMModuleNode::SaveToFile(const std::string& file_name, const std::string& format) {
  std::string fmt = runtime::GetFileFormat(file_name, format);
  std::error_code ecode;
#if TVM_LLVM_VERSION <= 70
  llvm::raw_fd_ostream dest(file_name, ecode, llvm::sys::fs::F_None);
#else
  llvm::raw_fd_ostream dest(file_name, ecode, llvm::sys::fs::OF_None);
#endif
  ICHECK_EQ(ecode.value(), 0) << "Cannot open file: " << file_name << " " << ecode.message();
  if (fmt == "o" || fmt == "obj") {
    With<LLVMTarget> llvm_target(*llvm_instance_, LLVMTarget::GetTargetMetadata(*module_));
#if TVM_LLVM_VERSION <= 60
    std::unique_ptr<llvm::Module> m = llvm::CloneModule(module_);
#else
    std::unique_ptr<llvm::Module> m = llvm::CloneModule(*module_);
#endif
    llvm::legacy::PassManager pass;
    llvm::TargetMachine* tm = llvm_target->GetOrCreateTargetMachine();
#if TVM_LLVM_VERSION <= 60
    ICHECK(tm->addPassesToEmitFile(pass, dest, llvm::TargetMachine::CGFT_ObjectFile) == 0)
        << "Cannot emit target CGFT_ObjectFile";
#elif TVM_LLVM_VERSION <= 90
    ICHECK(tm->addPassesToEmitFile(pass, dest, nullptr, llvm::TargetMachine::CGFT_ObjectFile) == 0)
        << "Cannot emit target CGFT_ObjectFile";
#else
    ICHECK(tm->addPassesToEmitFile(pass, dest, nullptr, llvm::CGFT_ObjectFile) == 0)
        << "Cannot emit target CGFT_ObjectFile";
#endif
    pass.run(*m);
  } else if (fmt == "s" || fmt == "asm") {
    With<LLVMTarget> llvm_target(*llvm_instance_, LLVMTarget::GetTargetMetadata(*module_));
#if TVM_LLVM_VERSION <= 60
    std::unique_ptr<llvm::Module> m = llvm::CloneModule(module_);
#else
    std::unique_ptr<llvm::Module> m = llvm::CloneModule(*module_);
#endif
    llvm::legacy::PassManager pass;
    llvm::TargetMachine* tm = llvm_target->GetOrCreateTargetMachine();
#if TVM_LLVM_VERSION <= 60
    ICHECK(tm->addPassesToEmitFile(pass, dest, llvm::TargetMachine::CGFT_AssemblyFile) == 0)
        << "Cannot emit target CGFT_AssemblyFile";
#elif TVM_LLVM_VERSION <= 90
    ICHECK(tm->addPassesToEmitFile(pass, dest, nullptr, llvm::TargetMachine::CGFT_AssemblyFile) ==
           0)
        << "Cannot emit target CGFT_AssemblyFile";
#else
    ICHECK(tm->addPassesToEmitFile(pass, dest, nullptr, llvm::CGFT_AssemblyFile) == 0)
        << "Cannot emit target CGFT_AssemblyFile";
#endif
    pass.run(*m);
  } else if (fmt == "ll") {
    module_->print(dest, nullptr);
  } else if (fmt == "bc") {
#if TVM_LLVM_VERSION <= 60
    llvm::WriteBitcodeToFile(module_, dest);
#else
    llvm::WriteBitcodeToFile(*module_, dest);
#endif
  } else {
    LOG(FATAL) << "Do not know how to save file " << file_name << " with format=\'" << format
               << "\'";
  }
  dest.close();
}

void LLVMModuleNode::SaveToBinary(dmlc::Stream* stream) {
  LOG(FATAL) << "LLVMModule: SaveToBinary not supported";
}

std::string LLVMModuleNode::GetSource(const std::string& format) {
  std::string fmt = runtime::GetFileFormat("", format);
  std::string type_str;
  llvm::SmallString<256> str;
  llvm::raw_svector_ostream rso(str);

  if (fmt == "s" || fmt == "asm") {
    With<LLVMTarget> llvm_target(*llvm_instance_, LLVMTarget::GetTargetMetadata(*module_));
#if TVM_LLVM_VERSION <= 60
    std::unique_ptr<llvm::Module> m = llvm::CloneModule(module_);
#else
    std::unique_ptr<llvm::Module> m = llvm::CloneModule(*module_);
#endif
    llvm::legacy::PassManager pass;
    llvm::TargetMachine* tm = llvm_target->GetOrCreateTargetMachine();
#if TVM_LLVM_VERSION <= 60
    ICHECK(tm->addPassesToEmitFile(pass, rso, llvm::TargetMachine::CGFT_AssemblyFile) == 0)
        << "Cannot emit target CGFT_AssemblyFile";
#elif TVM_LLVM_VERSION <= 90
    ICHECK(tm->addPassesToEmitFile(pass, rso, nullptr, llvm::TargetMachine::CGFT_AssemblyFile) == 0)
        << "Cannot emit target CGFT_AssemblyFile";
#else
    ICHECK(tm->addPassesToEmitFile(pass, rso, nullptr, llvm::CGFT_AssemblyFile) == 0)
        << "Cannot emit target CGFT_AssemblyFile";
#endif
    pass.run(*m);
    return rso.str().str();
  } else if (fmt == "" || fmt == "ll") {
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    ICHECK(module_ != nullptr);
    module_->print(rso, nullptr);
    return rso.str();
  } else {
    LOG(FATAL) << "Do not know how to get source code with format: " << format << "\'";
  }
  return "";
}

void LLVMModuleNode::Init(const IRModule& mod, const Target& target) {
  llvm_instance_ = std::make_unique<LLVMInstance>();
  With<LLVMTarget> llvm_target(*llvm_instance_, target);
  llvm::TargetMachine* tm = llvm_target->GetOrCreateTargetMachine();
  std::unique_ptr<CodeGenLLVM> cg = CodeGenLLVM::Create(llvm_target.get());

  std::vector<PrimFunc> funcs;
  std::string entry_func;
  relay::Runtime runtime =
      mod->GetAttr<relay::Runtime>(tvm::attr::kRuntime).value_or(relay::Runtime::Create("cpp"));
  bool system_lib = runtime->GetAttr<Bool>("system-lib").value_or(Bool(false));
  bool target_c_runtime = runtime->name == "crt";

  for (auto kv : mod->functions) {
    if (!kv.second->IsInstance<PrimFuncNode>()) {
      // (@jroesch): we relax constraints here, Relay functions will just be ignored.
      DLOG(INFO) << "Can only lower IR Module with PrimFuncs, but got " << kv.second->GetTypeKey();
      continue;
    }
    auto f = Downcast<PrimFunc>(kv.second);
    auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
    ICHECK(global_symbol.defined());
    function_names_.push_back(global_symbol.value());
    if (f->HasNonzeroAttr(tir::attr::kIsEntryFunc)) {
      entry_func = global_symbol.value();
    }
    funcs.push_back(f);
  }
  // TODO(@jroesch): follow up on this condition.
  // ICHECK(funcs.size() > 0);
  // TODO(tqchen): remove the entry function behavior as it does not
  // makes sense when we start to use multiple modules.
  cg->Init("TVMMod", llvm_target.get(), system_lib, system_lib, target_c_runtime);
  cg->SetFastMathFlags(llvm_target->GetFastMathFlags());

  cg->AddFunctionsOrdered(funcs.begin(), funcs.end());
  if (entry_func.length() != 0) {
    cg->AddMainFunction(entry_func);
  }

  module_owning_ptr_ = cg->Finish();
  module_ = module_owning_ptr_.get();
  llvm_target->SetTargetMetadata(module_);
  module_->addModuleFlag(llvm::Module::Override, "Debug Info Version",
                         llvm::DEBUG_METADATA_VERSION);

  if (tm->getTargetTriple().isOSDarwin()) {
    module_->addModuleFlag(llvm::Module::Override, "Dwarf Version", 2);
  }

  std::string verify_errors_storage;
  llvm::raw_string_ostream verify_errors(verify_errors_storage);
  LOG_IF(FATAL, llvm::verifyModule(*module_, &verify_errors))
      << "LLVM module verification failed with the following errors: \n"
      << verify_errors.str();
}

void LLVMModuleNode::Init(std::unique_ptr<llvm::Module> module,
                          std::unique_ptr<LLVMInstance> llvm_instance) {
  module_owning_ptr_ = std::move(module);
  module_ = module_owning_ptr_.get();
  llvm_instance_ = std::move(llvm_instance);
}

void LLVMModuleNode::LoadIR(const std::string& file_name) {
  auto llvm_instance = std::make_unique<LLVMInstance>();
  std::unique_ptr<llvm::Module> module = llvm_instance->LoadIR(file_name);
  Init(std::move(module), std::move(llvm_instance));
}

bool LLVMModuleNode::ImplementsFunction(const String& name, bool query_imports) {
  return std::find(function_names_.begin(), function_names_.end(), name) != function_names_.end();
}

void LLVMModuleNode::LazyInitJIT() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (ee_) {
    return;
  }
  With<LLVMTarget> llvm_target(*llvm_instance_, LLVMTarget::GetTargetMetadata(*module_));
  llvm::EngineBuilder builder(std::move(module_owning_ptr_));
  builder.setEngineKind(llvm::EngineKind::JIT);
  builder.setOptLevel(llvm::CodeGenOpt::Aggressive);
  builder.setMCPU(llvm_target->GetCPU());
  builder.setMAttrs(llvm_target->GetTargetFeatures());
  builder.setTargetOptions(llvm_target->GetTargetOptions());
  auto tm = std::unique_ptr<llvm::TargetMachine>(builder.selectTarget());
  if (!IsCompatibleWithHost(tm.get())) {
    LOG(FATAL) << "Cannot run module, architecture mismatch";
  }
  llvm::DataLayout layout(tm->createDataLayout());
  ICHECK(layout == module_->getDataLayout())
      << "Data layout mismatch between module("
      << module_->getDataLayout().getStringRepresentation() << ")"
      << " and ExecutionEngine (" << layout.getStringRepresentation() << ")";
  ee_ = builder.create(tm.release());
  ICHECK(ee_ != nullptr) << "Failed to initialize jit engine for " << module_->getTargetTriple();
  ee_->runStaticConstructorsDestructors(false);

  if (void** ctx_addr =
          reinterpret_cast<void**>(GetGlobalAddr(runtime::symbol::tvm_module_ctx, *llvm_target))) {
    *ctx_addr = this;
  }
  runtime::InitContextFunctions(
      [this, &llvm_target](const char* name) { return GetGlobalAddr(name, *llvm_target); });
  // There is a problem when a JITed function contains a call to a runtime function.
  // The runtime function (e.g. __truncsfhf2) may not be resolved, and calling it will
  // lead to a runtime crash.
  // Do name lookup on a symbol that doesn't exist. This will force MCJIT to finalize
  // all loaded objects, which will resolve symbols in JITed code.
  ee_->getFunctionAddress("__some_name_that_hopefully_doesnt_exist__b49f8aaade5877eaba7583b91");
}

bool LLVMModuleNode::IsCompatibleWithHost(const llvm::TargetMachine* tm) const {
  With<LLVMTarget> host_target(*llvm_instance_, "llvm");  // FIXME(kparzysz-quic): nesting
  auto tm_host = host_target->GetOrCreateTargetMachine();
  if (tm_host->getTargetTriple().getArch() != tm->getTargetTriple().getArch()) {
    LOG(INFO) << "Architecture mismatch: module=" << tm->getTargetTriple().str()
              << " host=" << tm_host->getTargetTriple().str();
    return false;
  }
  return true;
}

// Get global address from execution engine.
void* LLVMModuleNode::GetGlobalAddr(const std::string& name, const LLVMTarget& llvm_target) const {
  // first verifies if GV exists.
  if (module_->getGlobalVariable(name) != nullptr) {
    return reinterpret_cast<void*>(ee_->getGlobalValueAddress(name));
  } else {
    return nullptr;
  }
}

void* LLVMModuleNode::GetFunctionAddr(const std::string& name,
                                      const LLVMTarget& llvm_target) const {
  // first verifies if GV exists.
  if (module_->getFunction(name) != nullptr) {
    return reinterpret_cast<void*>(ee_->getFunctionAddress(name));
  } else {
    return nullptr;
  }
}

TVM_REGISTER_GLOBAL("target.build.llvm")
    .set_body_typed([](IRModule mod, Target target) -> runtime::Module {
      auto n = make_object<LLVMModuleNode>();
      n->Init(mod, target);
      return runtime::Module(n);
    });

TVM_REGISTER_GLOBAL("codegen.LLVMModuleCreate")
    .set_body_typed([](std::string target_str, std::string module_name) -> runtime::Module {
      auto llvm_instance = std::make_unique<LLVMInstance>();
      With<LLVMTarget> llvm_target(*llvm_instance, target_str);
      auto n = make_object<LLVMModuleNode>();
      // Generate a LLVM module from an input target string
      auto module = std::make_unique<llvm::Module>(module_name, *llvm_target->GetContext());
      llvm_target->SetTargetMetadata(module.get());
      module->setTargetTriple(llvm_target->GetTargetTriple());
      module->setDataLayout(llvm_target->GetOrCreateTargetMachine()->createDataLayout());
      n->Init(std::move(module), std::move(llvm_instance));
      return runtime::Module(n);
    });

TVM_REGISTER_GLOBAL("target.llvm_lookup_intrinsic_id")
    .set_body_typed([](std::string name) -> int64_t {
      return static_cast<int64_t>(llvm::Function::lookupIntrinsicID(name));
    });

TVM_REGISTER_GLOBAL("target.llvm_get_intrinsic_name").set_body_typed([](int64_t id) -> String {
#if TVM_LLVM_VERSION >= 130
  return std::string(llvm::Intrinsic::getBaseName(static_cast<llvm::Intrinsic::ID>(id)));
#elif TVM_LLVM_VERSION >= 40
  // This is the version of Intrinsic::getName that works for overloaded
  // intrinsics. Helpfully, if we provide no types to this function, it
  // will give us the overloaded name without the types appended. This
  // should be enough information for most uses.
  return std::string(llvm::Intrinsic::getName(static_cast<llvm::Intrinsic::ID>(id), {}));
#else
  // Nothing to do, just return the intrinsic id number
  return std::to_string(id);
#endif
});

TVM_REGISTER_GLOBAL("target.llvm_version_major").set_body_typed([]() -> int {
  return TVM_LLVM_VERSION / 10;
});

TVM_REGISTER_GLOBAL("runtime.module.loadfile_ll")
    .set_body_typed([](std::string filename, std::string fmt) -> runtime::Module {
      auto n = make_object<LLVMModuleNode>();
      n->LoadIR(filename);
      return runtime::Module(n);
    });

TVM_REGISTER_GLOBAL("codegen.llvm_target_enabled")
    .set_body_typed([](std::string target_str) -> bool {
      LLVMInstance llvm_instance;
      auto* tm = With<LLVMTarget>(llvm_instance, target_str)
                     ->GetOrCreateTargetMachine(/*allow_missing=*/true);
      return tm != nullptr;
    });

TVM_REGISTER_GLOBAL("codegen.codegen_blob")
    .set_body_typed([](std::string data, bool system_lib,
                       std::string llvm_target_string) -> runtime::Module {
      auto n = make_object<LLVMModuleNode>();
      auto llvm_instance = std::make_unique<LLVMInstance>();
      With<LLVMTarget> llvm_target(*llvm_instance, llvm_target_string);
      std::unique_ptr<llvm::Module> blob = CodeGenBlob(data, system_lib, llvm_target.get());
      n->Init(std::move(blob), std::move(llvm_instance));
      return runtime::Module(n);
    });

runtime::Module CreateLLVMCppMetadataModule(runtime::metadata::Metadata metadata, Target target,
                                            tvm::relay::Runtime runtime) {
  auto llvm_instance = std::make_unique<LLVMInstance>();
  With<LLVMTarget> llvm_target(*llvm_instance, target);
  bool system_lib = runtime->GetAttr<Bool>("system-lib").value_or(Bool(false));
  auto cg = std::make_unique<CodeGenCPU>();

  cg->Init("TVMMetadataMod", llvm_target.get(), system_lib, system_lib,
           /*target_c_runtime=*/false);

  cg->DefineMetadata(metadata);
  auto mod = cg->Finish();
  llvm_target->SetTargetMetadata(mod.get());
  mod->addModuleFlag(llvm::Module::Override, "Debug Info Version", llvm::DEBUG_METADATA_VERSION);

  if (llvm_target->GetOrCreateTargetMachine()->getTargetTriple().isOSDarwin()) {
    mod->addModuleFlag(llvm::Module::Override, "Dwarf Version", 2);
  }

  std::string verify_errors_storage;
  llvm::raw_string_ostream verify_errors(verify_errors_storage);
  LOG_IF(FATAL, llvm::verifyModule(*mod, &verify_errors))
      << "LLVM module verification failed with the following errors: \n"
      << verify_errors.str();

  auto n = make_object<LLVMModuleNode>();
  n->Init(std::move(mod), std::move(llvm_instance));

  auto meta_mod = MetadataModuleCreate(metadata);
  meta_mod->Import(runtime::Module(n));
  return meta_mod;
}

runtime::Module CreateLLVMCrtMetadataModule(const Array<runtime::Module>& modules, Target target,
                                            tvm::relay::Runtime runtime) {
  Array<String> func_names;
  for (runtime::Module mod : modules) {
    auto pf_funcs = mod.GetFunction("get_func_names");
    if (pf_funcs != nullptr) {
      Array<String> func_names_ = pf_funcs();
      for (const auto& fname : func_names_) {
        func_names.push_back(fname);
      }
    }
  }

  auto llvm_instance = std::make_unique<LLVMInstance>();
  With<LLVMTarget> llvm_target(*llvm_instance, target);
  bool system_lib = runtime->GetAttr<Bool>("system-lib").value_or(Bool(false));
  bool target_c_runtime = runtime->name == "crt";
  ICHECK(system_lib && target_c_runtime)
      << "For LLVM C-runtime metadata module, must include --system-lib and --runtime=c; "
      << "got target: " << target->str();
  auto cg = std::make_unique<CodeGenCPU>();
  cg->Init("TVMMetadataMod", llvm_target.operator->(), system_lib, system_lib, target_c_runtime);

  cg->DefineFunctionRegistry(func_names);
  auto mod = cg->Finish();
  llvm_target->SetTargetMetadata(mod.get());
  mod->addModuleFlag(llvm::Module::Override, "Debug Info Version", llvm::DEBUG_METADATA_VERSION);

  if (llvm_target->GetOrCreateTargetMachine()->getTargetTriple().isOSDarwin()) {
    mod->addModuleFlag(llvm::Module::Override, "Dwarf Version", 2);
  }

  std::string verify_errors_storage;
  llvm::raw_string_ostream verify_errors(verify_errors_storage);
  LOG_IF(FATAL, llvm::verifyModule(*mod, &verify_errors))
      << "LLVM module verification failed with the following errors: \n"
      << verify_errors.str();

  auto n = make_object<LLVMModuleNode>();
  n->Init(std::move(mod), std::move(llvm_instance));
  for (auto m : modules) {
    n->Import(m);
  }
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.CreateLLVMCrtMetadataModule")
    .set_body_typed(CreateLLVMCrtMetadataModule);

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
