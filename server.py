# Server code

import socket
import threading

PORT = 5050
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)

connections = {}  # Change to a dictionary to track connections by machine number
connected_machines = set()

def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")

    while True:
        data = conn.recv(1024).decode()
        if not data:
            break

        if data == "Disconnect":
            print(f"[DISCONNECT REQUEST] {addr} requested to disconnect.")
            machine_number = connections.pop(conn, None)
            if machine_number:
                connected_machines.remove(machine_number)
                update_active_connections()
                conn.close()
                print(f"[DISCONNECTED] {addr} disconnected.")
                return

        # Split received data into username and password
        username, password = data.split(',')
        machine_number = authenticate_user(username, password)

        if machine_number:
            if machine_number in connections.values():
                conn.send(f"Machine {machine_number} is already connected".encode())
                conn.close()
                return
            else:
                conn.send(f"Authenticated:{machine_number}".encode())  # Send machine number along with authentication message
                connections[conn] = machine_number
                connected_machines.add(machine_number)
                send_active_connections()
                # Once authenticated, keep the connection open until client closes it
                while True:
                    pass
        else:
            conn.send("Invalid credentials".encode())
            return

    print(f"[DISCONNECTED] {addr}")
    if conn in connections:
        machine_number = connections.pop(conn)
        connected_machines.remove(machine_number)
        update_active_connections()
    conn.close()

def authenticate_user(username, password):
    with open('user_credentials.txt', 'r') as file:
        for line in file:
            stored_username, stored_password, machine_number = line.strip().split(',')
            if username == stored_username and password == stored_password:
                return machine_number
    return None

def update_active_connections():
    print(f"[ACTIVE CONNECTIONS] {len(connections)}")
    print(f"[CONNECTED MACHINES] {', '.join(connected_machines)}")

def send_active_connections():
    update_active_connections()

def start():
    server.listen()
    print(f"[LISTENING] Server is listening on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()

print("[Starting] server is starting...")
start()
