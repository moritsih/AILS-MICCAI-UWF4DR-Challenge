import os


def main():
    # Determine the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Determine the project root directory (one level up from the script directory)
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    # Determine the appropriate command for the user's shell
    if os.name == 'nt':  # Windows
        if os.environ.get('TERM_PROGRAM', '').lower() == 'vscode' or 'powershell' in os.environ.get('PSMODULEPATH',
                                                                                                    '').lower():
            command = f'$env:PYTHONPATH="{project_root};$env:PYTHONPATH"'
        else:  # Default to Command Prompt if not detected as PowerShell
            command = f"set PYTHONPATH={project_root};%PYTHONPATH%"
    else:  # Unix-based (Linux, macOS)
        command = f"export PYTHONPATH={project_root}:$PYTHONPATH"

    # Print the command
    print("Run the following command in your terminal to set PYTHONPATH:")
    print(command)


if __name__ == "__main__":
    main()
