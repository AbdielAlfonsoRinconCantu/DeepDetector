import os

os.system('sudo ufw allow 39999:50000/udp')
print("hola")
os.system('sudo ./gev_nettweak eth0')

