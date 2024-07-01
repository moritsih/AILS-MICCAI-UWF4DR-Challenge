import os

def main():
    # Add the current working directory to PYTHONPATH
    current_dir = os.getcwd()

    # Determine the appropriate command for the user's shell
    if os.name == 'nt':  # Windows
        if os.environ.get('TERM_PROGRAM', '').lower() == 'vscode' or 'powershell' in os.environ.get('PSMODULEPATH', '').lower():
            command = f'$env:PYTHONPATH="{current_dir};$env:PYTHONPATH"'
        else:  # Default to Command Prompt if not detected as PowerShell
            command = f"set PYTHONPATH={current_dir};%PYTHONPATH%"
    else:  # Unix-based (Linux, macOS)
        command = f"export PYTHONPATH={current_dir}:$PYTHONPATH"

    # Print the command
    print("Run the following command in your terminal to set PYTHONPATH:")
    print(command)

if __name__ == "__main__":
    main()
