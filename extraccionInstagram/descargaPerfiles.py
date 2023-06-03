import os
from pathlib import Path
import random
import time
import logging
from PIL import Image
import instaloader
import json
from instaloader import Profile
from nima import nima
from email.message import EmailMessage
import ssl
import smtplib


PHOTOS_TO_DOWNLOAD = 30

nombreCuentas = ["username1","username2", "username3","username4","username5","username6","username7"]
password = ["password1","password2", "password3", "password4","password5","password6","password7"]

cuentasBloqueadas = []

userAgents =["Mozilla/5.0 (Linux; Android 7.0; SM-G920F Build/NRD90M; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/65.0.3325.109 Mobile Safari/537.36 Instagram 37.0.0.21.97 Android (24/7.0; 640dpi; 1440x2560; samsung; SM-G920F; zeroflte; samsungexynos7420; uk_UA; 98288242)", 
                "Mozilla/5.0 (Linux; Android 9; LG-H873 Build/PKQ1.190522.001; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/111.0.5563.57 Mobile Safari/537.36 Instagram 274.0.0.26.90 Android (28/9; 640dpi; 1440x2672; LGE/lge; LG-H873; lucye; lucye; en_CA; 456141812)" , 
                "Mozilla/5.0 (Linux; Android 12; RMX3241 Build/SP1A.210812.016; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/110.0.5481.153 Mobile Safari/537.36 Instagram 273.1.0.16.72 Android (31/12; 540dpi; 1080x2141; realme; RMX3241; RE513CL1; mt6833; it_IT; 455206193)", 
                "Mozilla/5.0 (Linux; Android 13; SM-A515F Build/TP1A.220624.014; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/110.0.5481.153 Mobile Safari/537.36 Instagram 274.0.0.26.90 Android (33/13; 480dpi; 1080x2168; samsung; SM-A515F; a51; exynos9611; it_IT; 456141810)" , 
                "Mozilla/5.0 (Linux; Android 13; SM-A536B Build/TP1A.220624.014; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/110.0.5481.153 Mobile Safari/537.36 Instagram 274.0.0.26.90 Android (33/13; 450dpi; 1080x2177; samsung; SM-A536B; a53x; s5e8825; en_GB; 456141791)", 
                "Mozilla/5.0 (Linux; Android 10; POT-LX1 Build/HUAWEIPOT-L21; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/110.0.5481.154 Mobile Safari/537.36 Instagram 275.0.0.0.47 Android (29/10; 408dpi; 1080x2139; HUAWEI; POT-LX1; HWPOT-H; kirin710; tr_TR; 455734489)",
                "Mozilla/5.0 (Linux; Android 12; SM-A325F Build/SP1A.210812.016; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/110.0.5481.153 Mobile Safari/537.36 Instagram 273.1.0.16.72 Android (31/12; 420dpi; 1080x2194; samsung; SM-A325F; a32; mt6769t; tr_TR; 455206193)"]

emailEmisor = 'miCorreo@gmail.com'
emailPassword = 'miContraseña'
emailReceptor = 'correoReceptor@gmail.com'

PERFILES = 'listaPerfiles'


#Esta función inicializa todas las cuentas que tenemos para
#la descarga de datos, y guarda los diferentes Objetos Instaloader en una lista
def initialize():
    lista = list()
    #Para cada cuenta, establecemos un user agent diferente. Si salta un error al hacer 
    #el login es porque la cuenta está bloqueada. Guardamos las cuentas bloqueadas en otra lista
    try:
        L1 = instaloader.Instaloader(download_videos=False,download_video_thumbnails=False, max_connection_attempts=1 , download_comments=True, save_metadata=True, post_metadata_txt_pattern="", iphone_support=False, user_agent=userAgents[1],slide='1')
        L1.login(nombreCuentas[1],password[1])
        lista.append(L1)
        logging.info("Acceso a cuenta 1 realizado correctamente")
    except:
        logging.error("Cuenta 1 bloqueada")
        cuentasBloqueadas.append(0)
        lista.append(L1)
    
    try:
        L2 = instaloader.Instaloader(download_videos=False,download_video_thumbnails=False, max_connection_attempts=1,download_comments=True, save_metadata=True, post_metadata_txt_pattern="", iphone_support=False, user_agent=userAgents[2],slide='1')
        L2.login(nombreCuentas[2],password[2])
        lista.append(L2)
        logging.info("Acceso a cuenta 2 realizado correctamente")
    except:
        logging.error("Cuenta 2 bloqueada")
        cuentasBloqueadas.append(1)
        lista.append(L2)

    try:
        L3 = instaloader.Instaloader(download_videos=False,download_video_thumbnails=False,max_connection_attempts=1,download_comments=True, save_metadata=True, post_metadata_txt_pattern="", iphone_support=False, user_agent=userAgents[3],slide='1')
        L3.login(nombreCuentas[3],password[3])
        lista.append(L3)
        logging.info("Acceso a cuenta 3 realizado correctamente")
    except:
        logging.error("Cuenta 3 bloqueada")
        cuentasBloqueadas.append(2)
        lista.append(L3)
   
    try:
        L4 = instaloader.Instaloader(download_videos=False,download_video_thumbnails=False,max_connection_attempts=1,download_comments=True, save_metadata=True, post_metadata_txt_pattern="", iphone_support=False, user_agent=userAgents[4],slide='1')
        L4.login(nombreCuentas[4],password[4])
        lista.append(L4)
        logging.info("Acceso a cuenta 4 realizado correctamente")
    except:
        logging.error("Cuenta 4 bloqueada")
        cuentasBloqueadas.append(3)
        lista.append(L4)

    try:
        L5 = instaloader.Instaloader(download_videos=False,download_video_thumbnails=False,max_connection_attempts=1,download_comments=True, save_metadata=True, post_metadata_txt_pattern="", iphone_support=False, user_agent=userAgents[5],slide='1')
        L5.login(nombreCuentas[5],password[5])
        lista.append(L5)
        logging.info("Acceso a cuenta 5 realizado correctamente")
    except:
        logging.error("Cuenta 5 bloqueada")
        cuentasBloqueadas.append(4)
        lista.append(L5)
    
    try:
        L6 = instaloader.Instaloader(download_videos=False,download_video_thumbnails=False,max_connection_attempts=1,download_comments=True, save_metadata=True, post_metadata_txt_pattern="", iphone_support=False, user_agent=userAgents[6],slide='1')
        L6.login(nombreCuentas[6],password[6])
        lista.append(L6)
        logging.info("Acceso a cuenta 6 realizado correctamente")
    except:
        logging.error("Cuenta 6 bloqueada")
        cuentasBloqueadas.append(5)
        lista.append(L6)

    return lista

#Devuelve la siguiente cuenta que debemos usar, evitando las cuentas bloqueadas
def siguienteCuenta(listaCuentas,index,blocked):
    sigCuenta = (index+1) % len(listaCuentas)
    if (len(blocked)!= len(listaCuentas)):
        while sigCuenta in blocked:
            sigCuenta = (sigCuenta+1) % len(listaCuentas)
    return sigCuenta

#Recuperamos información de los .json donde previamente habiamos guardado información
def checkpoint(nombreFichero):
    #Si existe el .json, recuperar los datos en forma de lista
    if os.path.exists(f"./"+nombreFichero+".json"):
        with open(f"./"+nombreFichero+".json",'r') as file:
                    return json.load(file)
    #Si no existe, crear el .json y devolver la lista vacia
    else:
        checkpoint = []
        save_checkpoint(nombreFichero,checkpoint)
        return checkpoint

#Guardamos la infromación actual en los json
def save_checkpoint(nombreFichero,datos):
    with open(f"./"+nombreFichero+".json",'w') as file:
        json.dump(datos,file,indent=4)

#Escoge la primera foto de un sidecar, devuelve -1 si solo hay videos
def first_photo(isVideoList):
    i = 0
    for isVideo in isVideoList:
        if not isVideo: return i
        i+=1
    return -1

def countPost(postType):
    list = checkpoint("postsType")
    if not list:
        list.append(0)
        list.append(0)
        list.append(0)
    if postType == "i":
        list[0]+=1
    elif postType == "s":
        list[1]+=1
    else:
        list[2]+=1
    save_checkpoint("postsType",list)

def makeStats():
    porc = []
    list = checkpoint("postsType")
    total = list[0]+list[1]+list[2]
    porc.append(round(list[0]/total*100,2))
    porc.append(round(list[1]/total*100,2))
    porc.append(round(list[2]/total*100,2))
    os.remove(f"./postsType.json")
    return porc

def removeUser(username): 
    list = checkpoint(PERFILES)
    list.remove(username)
    save_checkpoint(PERFILES,list)

def enviarCorreo(asunto):

    cuerpo = """
    Ha habido un error
    """
    em = EmailMessage()
    em['From'] = emailEmisor
    em['To'] = emailReceptor
    em['Subject'] = asunto
    em.set_content(cuerpo)
    
    contexto = ssl.create_default_context()

    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=contexto) as smtp:
        smtp.login(emailEmisor,emailPassword)
        smtp.sendmail(emailEmisor,emailReceptor,em.as_string())

    logging.info("Notificacion de error enviada")

def main():
    
    LEsp = instaloader.Instaloader(download_videos=False,download_video_thumbnails=False,max_connection_attempts=1,download_comments=True, save_metadata=False, post_metadata_txt_pattern="", iphone_support=False, user_agent=userAgents[0])
    LEsp.login(nombreCuentas[0],password[0])
    logging.info("Acceso a cuenta especial realizado correctamente")
    
    nerrors = 0
    nima_technical = nima.NimaScorer(tech=True)
    nima_aesthetic = nima.NimaScorer()

    #Creamos las diferentes sesiones de instaloader y las agrupamos en una lista
    listaCuentas = initialize()
    #Si hay muchas cunetas bloqueadas, esperar para empezar la recoleccion de datos
    while (len(cuentasBloqueadas)>=4):
        cuentasBloqueadas.clear()
        logging.error("Error = TooManyRequests-Grave")
        enviarCorreo("Muchas cuentas bloqueadas al ejecutar la aplicacion")
        logging.info("Esperaremos 2 horas")
        time.sleep(7200)
        listaCuentas = initialize()
    #Escogeremos la primera cuenta con la que empezar a descargar informacion
    cuentaEnUso = random.randint(0,len(listaCuentas)-1)
    while cuentaEnUso in cuentasBloqueadas:
            cuentaEnUso = random.randint(0,len(listaCuentas)-1)
    L = listaCuentas[cuentaEnUso]
    while True:
        try:
            logging.info(f"Cuenta {cuentaEnUso+1}")
            users = list()
            nperfiles = 0
            #Recuperamos la lista de todos los usernames que hemos recopidado
            usernames = checkpoint(PERFILES)
            #Recuperamos la lista de todos los usuarios que hemos descargado
            u = checkpoint("usuarios")
            perfilesDescargados = len(u)
            #Usernames contiene todos los usernames que habiamos recopilado pero todavia no hemos descargado
            usernames = usernames[perfilesDescargados:]
            usuarios = []
            nfotos = 1
            #Para cada username
            for username in usernames:
                #A partir del username cogemos el usuario de instagram correspondiente 
                logging.info(f"Perfil numero {perfilesDescargados+1} = {username}")
                try:
                    user = Profile.from_username(LEsp.context,username)
                except instaloader.exceptions.ProfileNotExistsException:
                    logging.warning(f"No se encuentra el perfil {username}")
                    removeUser(username)
                    continue
                if user.is_private: 
                    logging.warning(f"El perfil '{username}' ahora es privado")
                    removeUser(username)
                    continue
                if user.mediacount==0:
                    logging.warning(f"El perfil '{username}' no tiene ningun post")
                    removeUser(username)
                    continue

                perfilesDescargados+=1
                #De ese usuario creamos un iterador de todas las publicaciones que tiene
                posts = user.get_posts()
                publicaciones = []
                #Si ya estabamos a medias de recorrer ese iterador y ocurrió un error, volver al post en el que nos quedamos
                if os.path.exists(f"./resume_iterator.json"):
                    posts.thaw(instaloader.load_structure_from_file(LEsp.context,"resume_iterator.json"))
                    logging.debug("Recuperado")
                    #Recuperar la información de estos posts que ya habiamos descargado
                    publicaciones = checkpoint("posts")
                count = len(publicaciones)
                #Para cada publicacion
                for post in posts:
                    #Si es una foto o un carrusel de fotos (no videos)
                    if (post.typename!="GraphVideo"):
                        #Si todavia no hemos descargado suficientes fotos del usuario
                        if (count < PHOTOS_TO_DOWNLOAD):
                            #Cada 3 fotos descargadas cambiamos de cuenta (para evitar el error 429: Many Requests)
                            if (nfotos%5==0):
                                logging.info("Cambio de cuenta")
                                cuentaEnUso = siguienteCuenta(listaCuentas,cuentaEnUso,cuentasBloqueadas)
                                L = listaCuentas[cuentaEnUso]
                            logging.info(f"Descargando foto numero {count+1} con la cuenta {cuentaEnUso+1}")
                            #Descargar la foto y sus comentarios
                            sidSlide = 0
                            if (post.typename == "GraphSidecar"):
                                isVideoList = post.get_is_videos()
                                sidSlide = first_photo(isVideoList)
                                if sidSlide == -1: 
                                    logging.warning("Solo hay videos en este sidecar")
                                    countPost("s")
                                    continue
                                logging.debug(f"Cogemos el slide numero {sidSlide+1}")
                                L.slide_start = sidSlide
                                L.slide_end = sidSlide
                            L.download_post(post,Path("./descargas/"+user.username))
                            #Parar la ejecución unos segundos determinados (para evitar el error 429: Many Requests)
                            time.sleep(random.randint(10,12))
                            fecha = post.date.strftime("%Y-%m-%d_%H-%M-%S")
                            if (post.typename == "GraphImage"): filename = fecha+"_UTC.jpg"
                            else: filename = fecha+"_UTC_"+str(sidSlide+1)+".jpg"
                            image_path = f"./descargas/"+username+"/"+filename
                            with Image.open(image_path) as img:
                                score = nima_aesthetic.score(img)
                                tech_score = nima_technical.score(img)
                            os.remove(image_path)
                            #Guardar en una lista los datos importantes de la publicacion
                            publicaciones.append({
                                'type': post.typename,
                                'mediacount': post.mediacount,
                                'chosen_slide': sidSlide+1,
                                'score': score,
                                'tech_score': tech_score,
                                'caption': post.caption,
                                'date': fecha,
                                'likes':post.likes,
                                'comments_number':post.comments
                            })
                            save_checkpoint("posts",publicaciones)
                            if (post.typename == "GraphImage"): countPost("i")
                            else: countPost("s")
                            count+=1
                            nfotos+=1
                        else: break
                    else:
                        countPost("v")
                #Recuperamos la lista de todos los usuarios que hemos descargado
                usuarios = checkpoint("usuarios")
                #Añadimos el nuevo usuario que acabamos de descargar
                #firstpost = sorted(posts, key=lambda p:p.date)[0]
                #firstdate = firstpost.date.strftime("%Y/%m/%d, %H:%M:%S")
                listTypes = makeStats()
                usuarios.append({
                    'username': user.username,
                    'followers': user.followers,
                    'followees': user.followees,
                    'number_posts': user.mediacount,
                    'category': user._node["category_name"],
                    'image_porcentage': listTypes[0],
                    'sidecar_porcentage': listTypes[1],
                    'video_porcentage': listTypes[2],
                    'posts': publicaciones
                })
                #Lo guardamos en un .json
                save_checkpoint("usuarios",usuarios)
                #Eliminar la informacion de los posts de este usuario para poder empezar a guardar los del siguiente
                if os.path.exists(f"./posts.json"):
                    os.remove(f"./posts.json")
                #Eliminar la informacion de progreso del iterador
                if os.path.exists(f"./resume_iterator.json"):
                    os.remove(f"./resume_iterator.json")
            break            
        #Control de la excepción ManyRequests
        except instaloader.exceptions.TooManyRequestsException:
            #Guardar el progreso del iterador
            instaloader.save_structure_to_file(posts.freeze(),"resume_iterator.json")
            logging.debug("Guardado")
            #Añadir la cuenta a la lista de bloqueadas y parar un gran tiempo si es necesario
            cuentasBloqueadas.append(cuentaEnUso)
            if len(cuentasBloqueadas)>=4:
                cuentasBloqueadas.clear()
                x = random.randint(7200)
                logging.error("Error = TooManyRequests-Grave")
                enviarCorreo("Muchas cuentas bloqueadas (TooManyRequests)")
                logging.info("Esperaremos 2 horas")
            else:
                x = random.randint(1000,1200)
                logging.error("Error = TooManyRequests")
                logging.info(f"Esperaremos {x} segundos")
            time.sleep(x)
            #Cambiar a la siguiente cuenta y seguir descargando por donde nos quedamos 
            cuentaEnUso = siguienteCuenta(listaCuentas,cuentaEnUso,cuentasBloqueadas)
            L = listaCuentas[cuentaEnUso]
            logging.info("REANUDANDO")
        
        #Control de la excepción ConnectionException (fallo más común y que más ralentiza el proceso)
        except instaloader.exceptions.ConnectionException as err:
            #Guardar el progreso del iterador
            instaloader.save_structure_to_file(posts.freeze(),"resume_iterator.json")
            logging.debug("Guardado")
            #Añadir la cuenta a la lista de bloqueadas y parar un gran tiempo si es necesario
            cuentasBloqueadas.append(cuentaEnUso)
            if len(cuentasBloqueadas)>=4:
                cuentasBloqueadas.clear()
                x = random.randint(7000,7200)
                logging.error("Error = ConnectionException-Grave")
                enviarCorreo("Muchas cuentas bloqueadas (ConnectionException)")
                logging.info("Esperaremos 2 horas")
            else:
                x = random.randint(10,12)
                print(f"Ocurrió un error: {str(err)}")
                logging.exception("Error = ConnectionException")
                logging.info(f"Esperaremos {x} segundos")
            time.sleep(x)
            #Cambiar a la siguiente cuenta y seguir descargando por donde nos quedamos 
            cuentaEnUso = siguienteCuenta(listaCuentas,cuentaEnUso,cuentasBloqueadas)
            L = listaCuentas[cuentaEnUso]
            logging.info("REANUDANDO")
        
        #Guardar el progreso y parar el programa al hacer Control+C
        except KeyboardInterrupt as err:
            instaloader.save_structure_to_file(posts.freeze(),"resume_iterator.json")
            logging.debug("Guardado")
            logging.info("Parada manual del programa")
            break
        
        #Control de cualquier excepción esporádica
        except Exception as err:
            print(err)
            instaloader.save_structure_to_file(posts.freeze(),"resume_iterator.json")
            nerrors+=1
            logging.debug("Guardado")
            logging.error(f"Error = {type(err)}")
            enviarCorreo(f"Se ha producido un error ({type(err)})")
            if (nerrors>=15):
                logging.error("Demasiados errores")
                break
            logging.info("Esperaremos 30 segundos")
            time.sleep(30)
            cuentaEnUso = siguienteCuenta(listaCuentas,cuentaEnUso,cuentasBloqueadas)
            L = listaCuentas[cuentaEnUso]
            logging.info("REANUDANDO")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="log.log", filemode="w", format="%(asctime)s - %(levelname)s - %(message)s")
    main()