import socket

SERVER = socket.gethostbyname(socket.gethostname())
PORT = 5050
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

def send(msg):
    message = msg.encode()
    client.send(message)

def authenticate():
    username = input("Enter your username: ")
    password = input("Enter your password: ")
    credentials = f"{username},{password}"
    send(credentials)

    response = client.recv(1024).decode()
    print(response)
    if response.startswith("Authenticated"):
        return True
    else:
        return False

if authenticate():
    # Keep the connection open for further communication
    while True:
        pass
else:
    client.close()
    exit()
