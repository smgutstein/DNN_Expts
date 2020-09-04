import argparse
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
        print("Added: ", curr_file)

def git_commit():
    repo = Repo('.')
    repo.git.commit('-m','adding results to tfer to/from BIERS cluster')

def add_results_to_repo(root_dir):

    found_files = find_files(root_dir)
    print("Found Files:")
    for found_file in sorted(found_files):
        print("    ",found_file)


    subprocess.run(["git", "stash", "push"], check=True)
    subprocess.run(["git", "checkout", "result_branch"], check=True)
    git_add(found_files)
    git_commit()
    subprocess.run(["git", "checkout", "master"], check=True)
    subprocess.run(["git", "stash", "pop"], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get root dir for expt results')
    parser.add_argument('root_dir', type=str,
                         help='Get root dir for expt results')
    args = parser.parse_args()
    res_dir = args.root_dir

    try:
        add_results_to_repo(res_dir)
    except Exception as e:
        print("\nError adding results to repo. Returning to master branch")
        subprocess.run(["git", "checkout", "master"], check=True)
        print("\n\n")
        print(e)
        raise e
    
# python add_results_2_git.py ./results/opt_tfer_expts/tinyimagenet200_notliving_living_expts/wide_resnet_28_10_arch/tfer_nets
