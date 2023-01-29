import socket


class BotServer:
    def __init__(self, host, port, n_threads):
        self.host = host
        self.port = port
        self.listen = n_threads
        self.sock = None

    def create_sock(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, int(self.port)))
        self.sock.listen(int(self.listen))
        return self.sock

    def ready_for_client(self):
        # 연결 요청을 기다리다가 연결되면 리턴.
        return self.sock.accept()  # (sock_obj, cli_addr)

    def get_sock(self):
        return self.sock
