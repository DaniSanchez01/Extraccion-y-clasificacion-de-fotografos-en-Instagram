import os
import json

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

if __name__ == "__main__":
    
    MAX_PER_LIST = 1000
    NOMBRE_LISTA = 'listaPerfiles'

    nombresListas = ["perfilesHashtag1","perfilesHashtag2","perfilesHashtag3","perfilesHashtag4"]
    listasUsuarios = []

    for n in nombresListas:
        l = checkpoint(n)
        listasUsuarios.append(l["usernames"][:MAX_PER_LIST])
    i = 0
    listaFinal = checkpoint(NOMBRE_LISTA)
    while (i<MAX_PER_LIST):
        for lista in listasUsuarios:
            if (lista[i] not in listaFinal):
                listaFinal.append(lista[i])
            else:
                print("Usuario repetido: ",lista[i])
        i+=1
    save_checkpoint(NOMBRE_LISTA,listaFinal)
        
