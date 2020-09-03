from git import Repo
import os
import subprocess


def find_files(curr_dir, found_files=None):
    if found_files is None:
        found_files = []
    curr_sub_dirs = [x for x in os.listdir(curr_dir)
                     if os.path.isdir(os.path.join(curr_dir, x))]
    
    curr_found_files = [os.path.join(curr_dir,x)
                        for x in os.listdir(curr_dir)
                        if 'results.txt' == x or 'Expt_output.log' == x]

    found_files += curr_found_files

    for curr_sub_dir in curr_sub_dirs:
        found_files = find_files(os.path.join(curr_dir, curr_sub_dir),
                                 found_files)

    return found_files
    
    
def git_add(add_files):
    repo = Repo('.')
    for curr_file in add_files:
        repo.git.add('-f', curr_file)

def git_commit():
    repo = Repo('.')
    repo.git.commit('-m','adding results to tfer to/from BIERS cluster')

def add_results_to_repo():

    result_root_list = ['./results/opt_tfer_expts/cifar_100_living_living_expts/wide_resnet_28_10_arch/tfer_nets']

    for curr_result_root in result_root_list:
        found_files = find_files(curr_result_root)
        print("Found Files:")
        for found_file in sorted(found_files):
            print("    ",found_file)
        #git_add(found_files)

    subprocess.run(["git", "checkout", "result_branch"], check=True)
    #git_commit()
    subprocess.run(["git", "checkout", "master"], check=True)

if __name__ == "__main__":
    try:
        add_results_to_repo()
    except Exception as e:
        print("Error adding results to repo. Returning to master branch")
        subprocess.run(["git", "checkout", "master"], check=True)
        print("\n\n")
        print(e)
        raise e
    
