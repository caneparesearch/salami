#! /bin/env python
import os
import pandas as pd
from tqdm import tqdm

# pytest
result = os.system("pytest")
if result != 0:
    raise ValueError("pytest failed")

# make doc
api_doc_path = "docs/source/modules/"


def write_rst_for_sphinx(
    filename="pymatgen_cif.py",
    api_doc_path="docs/source/modules/",
    module="kmcpy.external",
    package="pymatgen_cif",
):
    with open(api_doc_path + filename.replace(".py", ".rst"), "w+") as rst:
        rst.write("""package
=========================

.. automodule:: modulename.package
    :members:
    :inherited-members:
                  """.replace("package", package).replace("modulename", module))
    return package


api_list = []

for root, dirs, files in os.walk("./salami", topdown=False):
    for name in files:
        filename = os.path.join(root, name)
        if (
            filename[-3:] == ".py"
            and ("__init__" not in filename)
            and ("_version" not in filename)
            and ("tools" not in root)
            and ("sites_and_lattices" not in filename)
            and ("obsolete" not in root)
            and ("test" not in root)
            and ("external" not in root)
        ):
            # for now, skip the tools
            # need to modify documentation

            print(root, name, " is python script")

            package = name.replace(".py", "")
            module_name = root.replace("./", "").replace("/", ".")

            write_rst_for_sphinx(
                filename=name,
                api_doc_path=api_doc_path,
                module=module_name,
                package=package,
            )
            api_list.append(package)

with open(api_doc_path + "api.rst", "w+") as api_file:
    filestring = """API Reference Documentation
===========================

.. toctree::
    :maxdepth: 2
    :caption: Contents:

"""
    for api in api_list:
        filestring += "    " + api + ".rst\n"
    api_file.write(filestring)
os.system("cd docs; make html; cd ..")

# remove dump
os.system("python rm_generator_dumps.py")

# check large files
unique_files = []
large_files = []
total_size = 0
os.system("git status > change_in_git")
os.system("git add -n . > git_to_be_added")
with open("change_in_git") as f:
    lines = f.readlines()
    for line in lines:
        filename = line.split("\t")[-1].replace("\n", "")
        # print(filename)
        if os.path.isfile(filename):
            megabyte = os.stat(filename).st_size / (1024 * 1024)
            total_size += megabyte
            # print(megabyte)
            if megabyte > 20:
                raise ValueError(f"{filename}larger than 20")

data = []
with open("git_to_be_added") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        filename = line.replace("add '", "").replace("'", "").replace("\n", "")
        # print(filename)
        if os.path.isfile(filename):

            megabyte = os.stat(filename).st_size / (1024 * 1024)
            total_size += megabyte
            print(filename, ":", megabyte, "Mb")
            data.append([filename, megabyte])
            if megabyte > 20:
                raise ValueError(f"{filename} larger than 20")
                large_files.append(filename)
if len(large_files) > 0:
    raise ValueError(large_files)
if total_size > 49:
    df = pd.DataFrame(data, columns=["filename", "size"])
    df.to_csv("filesize.csv")
    raise ValueError("commit larger than 49M may cause problem")
else:
    print("check done, all is not large file\n commit is smaller than 49Mb")
    print(f"total size is {total_size}Mb")


# commit

commit_message = input("please input commit message: ")
if len(commit_message) == 0:
    commit_message = "update"
os.system("git add .")
os.system(f'git commit -m "{commit_message}"')
os.system("git push")

# make container
# os.system("rm -rf /home/dx/repo/container/container/conda/salami.sif")
# os.system("sudo singularity build /home/dx/repo/container/container/conda/salami.sif /home/dx/repo/container/makefiles/Fornax/salami/oneapi.def")
# os.system("scp /home/dx/repo/container/container/conda/salami.sif fornax:/home/david/container")


# update


# pypi
# python -m build
