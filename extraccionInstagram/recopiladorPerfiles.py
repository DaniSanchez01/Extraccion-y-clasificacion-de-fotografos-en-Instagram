import os
from pathlib import Path
import random
import time
import instaloader
import json

####################################################################################################################
######                                         Parámetros a ajustar                                           ######
####################################################################################################################


accounts = ["username1","username2", "username3","username4","username5","username6","username7"]
passwords = ["password1","password2", "password3", "password4","password5","password6","password7"]

userAgents =[   "Mozilla/5.0 (Linux; Android 12; SM-A515F Build/SP1A.210812.016; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/109.0.5414.117 Mobile Safari/537.36 Instagram 269.0.0.18.75 Android (31/12; 420dpi; 1080x2186; samsung; SM-A515F; a51; exynos9611; it_IT; 444561847)" , 
                "Mozilla/5.0 (Linux; Android 12; SM-A025G Build/SP1A.210812.016; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/102.0.5005.125 Mobile Safari/537.36 Instagram 267.0.0.18.93 Android (31/12; 280dpi; 720x1471; samsung; SM-A025G; a02q; qcom; it_IT; 440638564)", 
                "Mozilla/5.0 (Linux; Android 13; SM-M127F Build/TP1A.220624.014; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/104.0.5112.97 Mobile Safari/537.36 Instagram 269.0.0.18.75 Android (33/13; 300dpi; 720x1465; samsung; SM-M127F; m12; exynos850; it_IT; 444561846)" , 
                "Mozilla/5.0 (Linux; Android 13; SM-G991B Build/TP1A.220624.014; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/109.0.5414.117 Mobile Safari/537.36 Instagram 269.0.0.18.75 Android (33/13; 420dpi; 1080x2320; samsung; SM-G991B; o1s; exynos2100; it_IT; 444561847)", 
                "Mozilla/5.0 (Linux; Android 12; RMX3521 Build/RKQ1.211119.001; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/109.0.5414.117 Mobile Safari/537.36 Instagram 269.0.0.18.75 Android (31/12; 480dpi; 1080x2153; realme; RMX3521; RE54E2L1; qcom; en_US; 444561847)",
                "Mozilla/5.0 (Linux; Android 12; SM-A127F Build/SP1A.210812.016; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/105.0.5195.136 Mobile Safari/537.36 Instagram 265.0.0.19.301 Android (31/12; 300dpi; 720x1467; samsung; SM-A127F; a12s; exynos850; it_IT; 436384411))"]

HASHTAG = "canon"
USERNAME_FILE = "perfilesCanon"
PROFILES_TO_DOWNLOAD = 2000

####################################################################################################################
####################################################################################################################

cuentasBloqueadas = []

def initialize():
    lista = list()
    
    try:
        L1 = instaloader.Instaloader(download_videos=False,download_video_thumbnails=False, max_connection_attempts=1 , download_comments=True, save_metadata=False, post_metadata_txt_pattern="", iphone_support=False, user_agent=userAgents[0])
        L1.login(accounts[0],passwords[0])
        lista.append(L1)
    except:
        print("Cuenta 1 bloqueada")
        cuentasBloqueadas.append(0)
        lista.append(L1)

    try:
        L2 = instaloader.Instaloader(download_videos=False,download_video_thumbnails=False, max_connection_attempts=1,download_comments=True, save_metadata=False, post_metadata_txt_pattern="", iphone_support=False, user_agent=userAgents[1])
        L1.login(accounts[1],passwords[1])
        lista.append(L2)
    except:
        print("Cuenta 2 bloqueada")
        cuentasBloqueadas.append(1)
        lista.append(L2)


    try:
        L3 = instaloader.Instaloader(download_videos=False,download_video_thumbnails=False,max_connection_attempts=1,download_comments=True, save_metadata=False, post_metadata_txt_pattern="", iphone_support=False, user_agent=userAgents[2])
        L1.login(accounts[2],passwords[2])
        lista.append(L3)
    except:
        print("Cuenta 3 bloqueada")
        cuentasBloqueadas.append(2)
        lista.append(L3)

    try:
        L4 = instaloader.Instaloader(download_videos=False,download_video_thumbnails=False,max_connection_attempts=1,download_comments=True, save_metadata=False, post_metadata_txt_pattern="", iphone_support=False, user_agent=userAgents[3])
        L1.login(accounts[3],passwords[3])
        lista.append(L4)
    except:
        print("Cuenta 4 bloqueada")
        cuentasBloqueadas.append(3)
        lista.append(L4)

    try:
        L5 = instaloader.Instaloader(download_videos=False,download_video_thumbnails=False,max_connection_attempts=1,download_comments=True, save_metadata=False, post_metadata_txt_pattern="", iphone_support=False, user_agent=userAgents[4])
        L1.login(accounts[4],passwords[4])
        lista.append(L5)
    except:
        print("Cuenta 5 bloqueada")
        cuentasBloqueadas.append(4)
        lista.append(L5)
   
    try:
        L6 = instaloader.Instaloader(download_videos=False,download_video_thumbnails=False,max_connection_attempts=1,download_comments=True, save_metadata=False, post_metadata_txt_pattern="", iphone_support=False, user_agent=userAgents[5])
        L1.login(accounts[5],passwords[5])
        lista.append(L6)
    except:
        print("Cuenta 6 bloqueada")
        cuentasBloqueadas.append(5)
        lista.append(L6)
  
    return lista

def siguienteCuenta(lista,index):
    
    sigCuenta = (index+1) % len(lista)
    while sigCuenta in cuentasBloqueadas:
        sigCuenta = (sigCuenta+1) % len(lista)
    return sigCuenta

def checkpoint(nombreFichero):
    #Si existe el .json, recuperar los datos en forma de lista
    if os.path.exists(f"./"+nombreFichero+".json"):
        with open(f"./"+nombreFichero+".json",'r') as file:
                    return json.load(file)
    #Si no existe, crear el .json y devolver la lista vacia
    else:
        checkpoint = {}
        checkpoint["usernames"] = []
        checkpoint["category"] = []
        save_checkpoint(nombreFichero,checkpoint)
        return checkpoint

#Guardamos la infromación actual en los json
def save_checkpoint(nombreFichero,datos):
    with open(f"./"+nombreFichero+".json",'w') as file:
        json.dump(datos,file,indent=4)


if __name__ == "__main__":

    listaCuentas = initialize()
    cuentaActual = random.randint(0,len(listaCuentas)-1)
    cuentaActual = siguienteCuenta(listaCuentas,cuentaActual)
    while True:
        try:
            L = listaCuentas[cuentaActual]
            print("Cambio a cuenta ",cuentaActual+1)
            #Cogemos un iterador que ira recorriendo los posts del hashtag que hayamos elegido 
            posts = instaloader.Hashtag.from_name(L.context, HASHTAG).get_posts_resumable()
            #Guardamos en una lista todos los perfiles que ya habiamos recopilado
            users = checkpoint(USERNAME_FILE)
            #Count guarda el numero de perfiles que faltan por recopilar (Num de perfiles que queremos recopilar - Num perfiles recopilados)
            count = PROFILES_TO_DOWNLOAD - len(users["usernames"])
            #Para cada post
            for post in posts:
                #Si pertenece a un usuario el cual no ha sido guardado todavia
                if (not post.owner_profile.username in users["usernames"]) and (count > 0):
                    #Guardamos el usuario y actualizamos count
                    p = instaloader.Profile.from_username(L.context,post.owner_profile.username)
                    time.sleep(random.randint(5,7))
                    users["usernames"].append(p.username)
                    users["category"].append(p._node["category_name"])
                    count-=1
                    print("Numero de perfiles encontrados =",len(users["usernames"]))
                    #Cada 20 perfiles encontrados se guardan en un .json (para no perderlos en caso de error) 
                    if (len(users["usernames"])%5==0):
                        save_checkpoint(USERNAME_FILE,users)
                    if (len(users["usernames"])%20==0):
                        cuentaActual = siguienteCuenta(listaCuentas,cuentaActual)
                        L = listaCuentas[cuentaActual]
                        print("Cambio a cuenta ",cuentaActual+1)




                else:
                    #Si ya hemos guardado todos los perfiles que queriamos, terminamos 
                    if count==0:
                        break
            break
        except instaloader.exceptions.ConnectionException as err:
            print("Error = ", type(err))
            cuentasBloqueadas.append(cuentaActual)
            if (len(cuentasBloqueadas)>=4):
                print("Demasiadas cuentas bloqueadas, esperamos media hora")
                time.sleep(1800)
                cuentasBloqueadas.clear()
                break
            cuentaActual = siguienteCuenta(listaCuentas,cuentaActual)
            print("ERROR: Esperaremos 10 segundos")
            time.sleep(10)
            print("REANUDANDO")
        except Exception as err:
            print("Error = ", type(err))
            print("ERROR: Esperaremos 10 segundos")
            time.sleep(10)
            print("REANUDANDO")
            cuentaActual = siguienteCuenta(listaCuentas,cuentaActual)




