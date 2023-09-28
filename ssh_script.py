import os
from pathlib import PurePosixPath, PureWindowsPath
import time
import paramiko

def get_last_runtime():
    """Returns the last runtime of this script."""
    with open("utils\\ssh_script_last_runtime.txt", 'r+') as f:
        last_runtime = float(f.read())
        # Update the last runtime of the script.
        f.seek(0)
        f.truncate()
        f.write(str(time.time()-5))  # Subtract 5s to account for potential runtime differences.

    return last_runtime

def sftp_upload_project(host, username, password, local_src, remote_dest):
    """Uploads all updated project files to the remote server.
    
    The files' last modified times are compared to the script's last runtime.
    """
    # Get the last runtime of the script.
    last_runtime = get_last_runtime()

    # Open a transport.
    transport = paramiko.Transport(host)
    # Authenticate.
    transport.connect(username=username, password=password)
    # Create an SFTP client.
    sftp = paramiko.SFTPClient.from_transport(transport)

    # Upload all local project files to the remote server.
    for root, dirs, files in os.walk(local_src):
        # Ignore hidden directories (i.e. git directories).
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for file in files:
            file_path = os.path.join(root, file)
            # Get the last modified time of the file.
            last_modified = os.path.getmtime(file_path)
            if last_modified > last_runtime:
                # NOTE: The remote server uses a Linux file system.
                relative_path = PurePosixPath(PureWindowsPath(os.path.relpath(file_path, local_src)))
                # Create the remote directory if it does not exist.
                remote_dir = str(PurePosixPath(remote_dest, os.path.dirname(relative_path)))
                try:
                    sftp.mkdir(remote_dir)
                except IOError:
                    pass
                # Upload the file.
                remote_path = str(PurePosixPath(remote_dest, relative_path))
                sftp.put(file_path, remote_path)

    # Close the transport.
    sftp.close()
    transport.close()

def get_script_relative_path(local_src, script_name):
    """Returns the relative path to the script to be run remotely.

    Since the project folder in the remote server is a duplicate of the local 
    project folder, the local relative path to the script is the same as that 
    of the remote server.

    The shell session starts in /home/tim so the relative path will need to 
    include the project folder name.
    """
    local_path = None
    # Find the path to the script on the local machine.
    for root, dirs, files in os.walk(local_src):
        # Ignore hidden directories (i.e. git directories).
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for file in files:
            if file == script_name:
                local_path = os.path.join(root, file)
                break
    # Check that the script was found.
    if not local_path:
        raise FileNotFoundError(f"Could not find {script_name}")

    # Get the relative path to the script to be run remotely.
    # Get the project folder name.
    project_folder = os.path.basename(os.path.normpath(local_src))
    # NOTE: The remote server uses a Linux file system.
    relative_path = str(PurePosixPath(PureWindowsPath(project_folder, os.path.relpath(local_path, local_src))))

    return relative_path

def ssh_run_script(host, username, password, local_src, script_name):
    """Runs a script on the remote server."""
    # Get the relative path to the script.
    relative_path = get_script_relative_path(local_src, script_name)
    # Add a backslash before all whitespace characters so that the full path 
    # is not interpreted as multiple arguments.
    relative_path = relative_path.replace(" ", "\ ")

    # Connect to the SSH server.
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=username, password=password)

    # Run the script.
    command = f"python3 {relative_path}"
    stdin, stdout, stderr = client.exec_command(command)

    # Print the output.
    if stdout.channel.recv_exit_status() == 0:
        print(f"STDOUT: {stdout.read().decode('utf-8')}")
    else:
        print(f"STDERR: {stderr.read().decode('utf-8')}")

    # Close the connection.
    stdin.close()
    stdout.close()
    stderr.close()
    client.close()



if __name__ == "__main__":
    # SSH login info.
    host = "10.167.60.43"
    username = "tim"
    password = "Spider"

    # Upload directories.
    local_src = "D:\\Uni\\Yessir, its a Thesis\\SNN Seizure Detection"
    remote_dest = "/home/tim/SNN Seizure Detection"

    # Python script to be run remotely.
    script_name = "slice_tuh_data.py"

    # Update the project files on the remote server.
    sftp_upload_project(host, username, password, local_src, remote_dest)
    # Run the chosen script on the remote server.
    ssh_run_script(host, username, password, local_src, script_name)