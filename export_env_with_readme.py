import subprocess
import json
import os

def get_conda_env_info():
    # Run the conda list command and capture the output
    result = subprocess.run(['conda', 'list', '--json'], stdout=subprocess.PIPE, text=True)
    
    # Parse the JSON output
    packages = json.loads(result.stdout)
    
    # Prepare environment.yml content
    conda_packages = []
    for package in packages:
        conda_packages.append(f"{package['name']}={package['version']}")
    return conda_packages

def get_python_package_versions():
    # Use the pip list command to get installed Python packages and their versions
    result = subprocess.run(['pip', 'list', '--format=json'], stdout=subprocess.PIPE, text=True)
    
    # Parse the JSON output
    packages = json.loads(result.stdout)
    
    # Prepare requirements.txt content
    pip_packages = []
    for package in packages:
        pip_packages.append(f"{package['name']}=={package['version']}")
    return pip_packages

def save_environment_files(conda_packages, pip_packages, folder_name, env_name):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Save environment.yml
    with open(os.path.join(folder_name, f'{env_name}_environment.yml'), 'w') as file:
        file.write(f"name: {env_name}\n")
        file.write("channels:\n")
        file.write("  - defaults\n")
        file.write("dependencies:\n")
        for package in conda_packages:
            file.write(f"  - {package}\n")
        file.write("  - pip:\n")
        for package in pip_packages:
            file.write(f"    - {package}\n")

    # Save requirements.txt
    with open(os.path.join(folder_name, f'{env_name}_requirements.txt'), 'w') as file:
        for package in pip_packages:
            file.write(f"{package}\n")

def create_readme(folder_name, env_name):
    with open(os.path.join(folder_name, 'README.txt'), 'w') as file:
        file.write(f"Commands to create the environment '{env_name}':\n")
        file.write("1. Create the conda environment:\n")
        file.write(f"   conda create -n {env_name} python=3.8\n")
        file.write("2. Activate the conda environment:\n")
        file.write(f"   conda activate {env_name}\n")
        file.write("3. Install conda packages:\n")
        file.write(f"   conda env update --file {env_name}_environment.yml --prune\n")
        file.write("4. Install pip packages:\n")
        file.write(f"   pip install -r {env_name}_requirements.txt\n")

def main():
    env_name = input("Enter the name of the environment: ")
    folder_name = f"{env_name}_HowToCreate"

    print("Collecting Conda Environment Packages...")
    conda_packages = get_conda_env_info()
    print("Collecting Python Packages...")
    pip_packages = get_python_package_versions()

    print("Saving environment files...")
    save_environment_files(conda_packages, pip_packages, folder_name, env_name)

    print("Creating README file...")
    create_readme(folder_name, env_name)

    print(f"Files have been saved in the folder '{folder_name}'.")

if __name__ == "__main__":
    main()
