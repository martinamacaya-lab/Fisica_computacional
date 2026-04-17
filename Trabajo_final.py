import numpy as np
import matplotlib.pyplot as plt

########### Definición de los parametros y variables ###########

#Parámetros del pulso S
f0 = 2e6 #Frecuencia central del pulso (Hz)
A = 1 #Amplitud maxima del pulso
tc = 2/f0 #Tiempo central
sigma = 0.5/f0 #Ancho de la campana de Gauss, duracion pulso
xs_idx = 1 #indice donde se pone el ultrasonido a generar el pulso

#Parametros del medio
c1 = 1540 #velocidad del medio 1, es un tejido
c2 = 3000 #velocidad del medio 2, es un hueso
alpha_max = 5e7 #Valor maximo que puede tomar alpha en el ultimo nodo

#Variables del dominio espacial
L = 0.05 #Longitud del medio
Nx = 1000 #Número de puntos (nodos) espaciales
x = np.linspace(0, L, Nx) #arreglo con la cantidad de nodos
dx = x[1] - x[0] #Delta x

#Variables del dominio temporal
#Determinar el valor para tener estabilidad, Courant 
c_max = max(c1, c2)  # Buscamos la velocidad más rápida del sistema
Courant = 0.5   # Factor de seguridad (debe ser <= 1)
dt = Courant * dx / c_max  # Tamaño del paso de tiempo calculado para ser estable

T = 5e-5  # Tiempo total de simulación
Nt = int(round(T / dt)) + 1   # Cantidad total de pasos en el bucle temporal
t = np.linspace(0, Nt*dt, Nt) # Creación del arreglo de tiempo discreto


########## Creación del medio heterogéneo (malla) #################
#Arreglos de ceros
c = np.zeros(Nx) #para guardar la velocidad en cada nodo
alpha = np.zeros(Nx) #guardar la atenuación en cada nodo

#Definir el punto exacto de la interfaz entre tejidos
indice_interfaz = Nx // 2

#Definir las velocidades en cada medio
c[:indice_interfaz] = c1 #Los nodos donde la velocidad es del medio 1 lado izquierdo
c[indice_interfaz:] = c2 #Los nodos donde la velocidad es del medio 2 lado derecho

#Definir la atenuación para el medio 2 
nodos_medio_2 = Nx - indice_interfaz #calcular la cantidad de nodos tiene el medio 2
perfil_crecimiento = np.linspace(0, 1, nodos_medio_2) ** 2 #la atenuación tiene un crecimiento suave se genera un vector
alpha[indice_interfaz:] = alpha_max * perfil_crecimiento


############### Definición fuente del pulso ##########
#Calcular S
S = A * np.sin(2 * np.pi * f0 * t) * np.exp(-1 * ((t - tc) / (2 * sigma))**2)


############### Condiciones iniciales y matrices vacias ###############
#Desplazamiento inicial nulo y velocidad inicial nula
u_previo = np.zeros(Nx) #Desplazamiento en el tiempo n-1
u_actual = np.zeros(Nx) #Desplazamiento en el tiempo n
u_nuevo = np.zeros(Nx) #Desplazamiento en el tiempo n+1


############ Aplicar metodo de diferencias finitas #########################################

#Realizar el ciclo for para el tiempo
for n in range(Nt):

    #Realizar el ciclo for para el espacio
    for i in range(1, Nx-1):
        #Calcular velocidades en la interfaz para que no tenga un salto brusco (Promediar)
        c2_i_izquierda = 0.5 * (c[i]**2 + c[i-1]**2) #c^2_{i-1/2}
        c2_i_derecha = 0.5 * (c[i]**2 + c[i+1]**2) #c^2_{i+1/2}
        
        #Calcular la derivada relacionada con el coeficiente variable
        derivada_ceoficiente = (c2_i_derecha * (u_actual[i+1] - u_actual[i]) - c2_i_izquierda * (u_actual[i] - u_actual[i-1])) / dx**2
   
        #Derivada de atenuación despues de despejar y definición del alpha 
        alpha_i = alpha[i] #alpha en el nodo i
        denominador_alpha = 1 + alpha_i * (dt / 2) 
        multiplicador_alpha = 1 - alpha_i * (dt / 2) 

        #ecuación de onda final discretizada para el nodo i
        u_nuevo[i] = (1 / denominador_alpha) * (2 * u_actual[i] - multiplicador_alpha * u_previo[i] + dt**2 * derivada_ceoficiente)


    #Implementar la fuente del pulso en el nodo específico
    u_nuevo[xs_idx] = u_nuevo[xs_idx] + S[n] * dt**2 

    ##Definir las condiciones de borde##

    #Condición de borde izquierdo x=0 (condición de Sommerfeld)
    u_nuevo[0] = u_actual[0] + (c[0] * (dt / dx))*(u_actual[1] - u_actual[0])
    #Condición de borde derecho x=L (Dirichlet)
    u_nuevo[-1] = 0

    #Ir guardando los desplazamientos para el siguiente ciclo
    u_previo[:] = u_actual[:] #lo que era presente ahora es el pasado
    u_actual[:] = u_nuevo[:] #lo que era futuro ahora es el presente

