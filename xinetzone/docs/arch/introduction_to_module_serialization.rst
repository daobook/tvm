..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

模块序列化简介
====================================

部署 TVM 运行时模块时，无论是 CPU 还是 GPU, TVM 只需要一个动态共享库即可。关键是我们统一的模块序列化机制。本文档将介绍 TVM 模块序列化格式标准及实现细节。

*********************
Module 导出示例
*********************

以 GPU 构建 ResNet-18 工作负载作为例子。

.. code:: python

   from tvm import relay
   from tvm.relay import testing
   from tvm.contrib import utils
   import tvm

   # Resnet18 workload
   resnet18_mod, resnet18_params = relay.testing.resnet.get_workload(num_layers=18)

   # build
   with relay.build_config(opt_level=3):
       _, resnet18_lib, _ = relay.build_module.build(resnet18_mod, "cuda", params=resnet18_params)

   # create one tempory directory
   temp = utils.tempdir()

   # path lib
   file_name = "deploy.so"
   path_lib = temp.relpath(file_name)

   # export library
   resnet18_lib.export_library(path_lib)

   # load it back
   loaded_lib = tvm.runtime.load_module(path_lib)
   assert loaded_lib.type_key == "library"
   assert loaded_lib.imported_modules[0].type_key == "cuda"

*************
序列化
*************

入口 API 是 ``tvm.module.Module`` 的 ``export_library`` 。在这个函数中，将执行以下步骤：

1. 收集所有 DSO 模块（LLVM 模块和 C 模块）
2. 一旦有了 DSO 模块，将调用 ``save`` 函数将它们保存到文件中。
3. 接下来，将检查是否导入了模块，如 CUDA, OpenCL 或其他任何东西。
   这里不限制模块类型。导入模块后，将创建名为 ``devc.o`` / ``dev.cc`` 的文件（这样就可以将导入模块的二进制 blob 数据嵌入到动态共享库中），
   然后调用函数 ``_PackImportsToLLVM`` 或 ``_PackImportsToC`` 来进行模块序列化。
4. 最后，回调 ``fcompile``，它调用 ``_cc.create_shared`` 获取动态共享库。

.. note::
    1. 对于 C 源码模块，我们将编译它们，并将它们与 DSO 模块链接在一起。

    2. 使用 ``_PackImportsToLLVM`` 或 ``_PackImportsToC`` 取决于我们是否在 TVM 中启用 LLVM。
      他们实际上达到了相同的目标。


***************************************************
在序列化和格式标准的框架下
***************************************************

如前所述，将在 ``_PackImportsToLLVM`` 或 ``_PackImportsToC`` 中进行序列化工作。
它们都调用 ``SerializeModule`` 来序列化运行时模块。在 ``SerializeModule`` 函数中，首先构造了辅助类 ``ModuleSerializer``。
它将需要模块做一些初始化工作，如标记模块索引。然后可以使用它的 ``SerializeModule`` 来序列化模块。

为了更好地理解，更深入地研究这个类的实现。

下面的代码用于构造 ``ModuleSerializer``：

.. code:: c++

   explicit ModuleSerializer(runtime::Module mod) : mod_(mod) {
     Init();
   }
   private:
   void Init() {
     CreateModuleIndex();
     CreateImportTree();
   }

在 ``CreateModuleIndex()`` 中，将使用 DFS 检查模块导入关系，并为它们创建索引。注意，root 模块固定在位置 0。在我们的例子中，有这样的模块关系：

.. code:: c++

  llvm_mod:imported_modules
    - cuda_mod

所以 LLVM 模块的索引为 0，CUDA 模块的索引为 1。

在构造模块索引之后，尝试构造导入树（``CreateImportTree()``），当重新加载导出的库时，它将用于恢复模块导入关系。
在我们的设计中，使用 CSR 格式来存储导入树，每一行是父索引，child 索引对应其 children 索引。
在代码中，使用 ``import_tree_row_ptr_`` 和 ``import_tree_child_indices_`` 来表示它们。

在初始化之后，可以使用 ``SerializeModule`` 函数来序列化模块。在它的函数逻辑中，将假设序列化格式如下：

.. code:: c++

   binary_blob_size
   binary_blob_type_key
   binary_blob_logic
   binary_blob_type_key
   binary_blob_logic
   ...
   _import_tree
   _import_tree_logic

``binary_blob_size`` 是这个序列化步骤中 blob 的数量。在我们的例子中有三个 blob，分别为 LLVM 模块、CUDA 模块和 ``_import_tree`` 创建。

``binary_blob_type_key`` 是模块的 blob 类型键。对于 LLVM / C 模块，其 blob 类型键为 ``_lib``。
对于 CUDA 模块，它是 ``cuda``，可以通过 ``module->type_key()`` 获取。

``binary_blob_logic`` 是对 blob 的逻辑处理。对于大多数 blob（如 CUDA, OpenCL），我们将调用 ``SaveToBinary`` 函数将 blob 序列化为二进制。
然而，像 LLVM / C 模块一样，将只写 ``_lib`` 来表明这是 DSO 模块。

.. note::
   是否需要实现 ``SaveToBinary`` 虚函数（virtual function）取决于模块的使用方式。
   例如，如果模块中有我们在加载动态共享库时需要的信息，我们应该这样做。
   与 CUDA 模块一样，我们在加载动态共享库时需要将其二进制数据传递给 GPU 驱动，因此我们需要实现 ``SaveToBinary`` 对其二进制数据进行序列化。
   但是对于主机模块（如 DSO），在加载动态共享库时不需要其他信息，因此不需要实现 ``SaveToBinary``。
   但是，如果将来我们想要记录 DSO 模块的一些元信息，我们也可以为 DSO 模块实现 ``SaveToBinary``。

最后，我们将写入一个键 ``_import_tree`` ，除非我们的模块只有一个 DSO 模块并且它位于根目录中。
当我们像前面说的那样重新加载导出的库时，它被用来重建模块导入关系。
``import_tree_logic`` 只是将 ``import_tree_row_ptr_`` 和 ``import_tree_child_indices_`` 写入流。

在这一步之后，将把它打包到可以在动态库中恢复的 symbol ``runtime::symbol::tvm_dev_mblob`` 中。

现在，完成了序列化部分。如您所见，理想情况下，可以支持导入任意模块。

****************
反序列化
****************

入口 API 是 ``tvm.runtime.load``。这个函数实际上是调用 ``_LoadFromFile``。
如果再深入一点，这就是 ``Module::LoadFromFile``。在示例中，该文件是 ``deploy.so``。
因此，根据函数逻辑，将在 ``dso_library.cc`` 中回调 ``module.loadfile_so``。关键点在：

.. code:: c++

   // Load the imported modules
   const char* dev_mblob = reinterpret_cast<const char*>(lib->GetSymbol(runtime::symbol::tvm_dev_mblob));
   Module root_mod;
   if (dev_mblob != nullptr) {
   root_mod = ProcessModuleBlob(dev_mblob, lib);
   } else {
   // Only have one single DSO Module
   root_mod = Module(n);
   }

如前所述，把这个 blob 打包到 symbol ``runtime::symbol::tvm_dev_mblob`` 中。
在反序列化部分，我们将检查它。如果我们有 ``runtime::symbol::tvm_dev_mblob``，我们将调用 ``ProcessModuleBlob``，其逻辑如下所示：

.. code:: c++

   READ(blob_size)
   READ(blob_type_key)
   for (size_t i = 0; i < blob_size; i++) {
       if (blob_type_key == "_lib") {
         // construct dso module using lib
       } else if (blob_type_key == "_import_tree") {
         // READ(_import_tree_row_ptr)
         // READ(_import_tree_child_indices)
       } else {
         // call module.loadbinary_blob_type_key, such as module.loadbinary_cuda
         // to restore.
       }
   }
   // Using _import_tree_row_ptr and _import_tree_child_indices to
   // restore module import relationship. The first module is the
   // root module according to our invariance as said before.
   return root_module;

在此之后，把 ``ctx_address`` 设置为 ``root_module``，以便允许从 root 查找 symbol（这样所有 symbol 都是可见的）。

最终，完成反序列化部分。
