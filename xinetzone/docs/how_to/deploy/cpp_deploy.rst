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

使用 C++ API 部署 TVM Module
===============================

`apps/howto_deploy <https://github.com/apache/tvm/tree/main/apps/howto_deploy>`_ 中提供了如何部署 TVM 模块的示例，可以使用以下命令运行该示例：


.. code:: bash

    cd apps/howto_deploy
    ./run_example.sh


获取 TVM 运行时库
-----------------------

The only thing we need is to link to a TVM runtime in your target platform.
TVM provides a minimum runtime, which costs around 300K to 600K depending on how much modules we use.
In most cases, we can use ``libtvm_runtime.so`` that comes with the build.

If somehow you find it is hard to build ``libtvm_runtime``, checkout
`tvm_runtime_pack.cc <https://github.com/apache/tvm/tree/main/apps/howto_deploy/tvm_runtime_pack.cc>`_.
It is an example all in one file that gives you TVM runtime.
You can compile this file using your build system and include this into your project.

You can also checkout `apps <https://github.com/apache/tvm/tree/main/apps/>`_ for example applications build with TVM on iOS, Android and others.

Dynamic Library vs. System Module
---------------------------------
TVM provides two ways to use the compiled library.
You can checkout `prepare_test_libs.py <https://github.com/apache/tvm/tree/main/apps/howto_deploy/prepare_test_libs.py>`_
on how to generate the library and `cpp_deploy.cc <https://github.com/apache/tvm/tree/main/apps/howto_deploy/cpp_deploy.cc>`_ on how to use them.

- Store library as a shared library and dynamically load the library into your project.
- Bundle the compiled library into your project in system module mode.

Dynamic loading is more flexible and can load new modules on the fly. System module is a more ``static`` approach.  We can use system module in places where dynamic library loading is banned.
