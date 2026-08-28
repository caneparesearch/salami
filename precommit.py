#! /bin/env python
import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm

def run_pytest():
    result = os.system("pytest")
    if result != 0:
        raise ValueError("pytest failed")

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

def update_docs():
    api_doc_path = "docs/source/modules/"
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
    os.system("python rm_generator_dumps.py")

def update_git():
    unique_files = []
    large_files = []
    total_size = 0
    os.system("git status > change_in_git")
    os.system("git add -n . > git_to_be_added")
    
    with open("change_in_git") as f:
        lines = f.readlines()
        for line in lines:
            filename = line.split("\t")[-1].replace("\n", "")
            if os.path.isfile(filename):
                megabyte = os.stat(filename).st_size / (1024 * 1024)
                total_size += megabyte
                if megabyte > 20:
                    raise ValueError(f"{filename}larger than 20")

    data = []
    with open("git_to_be_added") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            filename = line.replace("add '", "").replace("'", "").replace("\n", "")
            if os.path.isfile(filename):
                megabyte = os.stat(filename).st_size / (1024 * 1024)
                total_size += megabyte
                print(filename, ":", megabyte, "Mb")
                data.append([filename, megabyte])
                if megabyte > 20:
                    large_files.append(filename)
                    
    if len(large_files) > 0:
        raise ValueError(f"Files larger than 20MB detected: {large_files}")
        
    if total_size > 49:
        df = pd.DataFrame(data, columns=["filename", "size"])
        df.to_csv("filesize.csv")
        raise ValueError("commit larger than 49M may cause problem")
    else:
        print("check done, all is not large file\n commit is smaller than 49Mb")
        print(f"total size is {total_size}Mb")

    commit_message = input("please input commit message: ")
    if len(commit_message) == 0:
        commit_message = "update"
    os.system("git add .")
    os.system(f'git commit -m "{commit_message}"')
    os.system("git push")

def update_container():
    os.system("rm -rf /home/dx/repo/container/container/python/salami.sif")
    os.system(
        "cd /home/dx/repo/container && sudo singularity build  /home/dx/repo/container/container/python/salami.sif /home/dx/repo/container/makefiles/NUS_HPC/salami/salami.def; cd -"
    )
    input("please check the container and press enter to continue, or ctrl+c to stop")
    os.system("ssh archer rm /mnt/lustre/a2fs-work2/work/e05/e05/xiewz/container/salami.sif")
    os.system("scp /home/dx/repo/container/container/python/salami.sif archer:/mnt/lustre/a2fs-work2/work/e05/e05/xiewz/container/")

def main():
    parser = argparse.ArgumentParser(description="Repository automation script")
    parser.add_argument("-t", "--test", action="store_true", help="Run pytest")
    parser.add_argument("-d", "--doc", action="store_true", help="Update documentation")
    parser.add_argument("-c", "--container", action="store_true", help="Update container")
    parser.add_argument("-g", "--git", action="store_true", help="Update git repository")
    parser.add_argument("-a", "--all", action="store_true", help="Run all tasks sequentially")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    run_all = args.all

    if run_all or args.test:
        run_pytest()
    if run_all or args.doc:
        update_docs()
    if run_all or args.git:
        update_git()
    if run_all or args.container:
        update_container()

if __name__ == "__main__":
    main()
    