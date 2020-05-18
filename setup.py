# Copyright (C) IBM Corporation 2020
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="visual_reasoning_nemo_collection",
    version="0.0.1",
    author="Alexis Asseman",
    author_email="",
    description="Collection of reusable NeMo modules to support the visual-reasoning models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IBM/visual-reasoning",
    packages=['visual_reasoning_nemo_collection'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'nemo_toolkit' # Actually depending on the latest master as of 2020-05-12. Will add version when possible.
    ],
)
