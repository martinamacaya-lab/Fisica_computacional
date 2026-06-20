import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

########### Definición de los parametros y variables ###########

#Parámetros del pulso S
f0 = 2e6 #Frecuencia central del pulso (Hz)
A = 1e6 #Amplitud maxima del pulso
tc = 2/f0 #Tiempo central
sigma = 0.5/f0 #Ancho de la campana de Gauss, duracion pulso
xs_idx = 1 #indice donde se pone el ultrasonido a generar el pulso

#Parametros de cada medio
c1 = 1540 #velocidad del medio 1, es un tejido blando
c2 = 3500 #velocidad del medio 2, es un hueso craneal

#Variables del dominio espacial
cuero_cabelludo = 8e-3  # 8 mm
periostio = (20 + 80 + 80) * 1e-6  # 180 µm
L1 = cuero_cabelludo + periostio  # Medio 1, es Tejido blando de 0.00818 m
L2 = 7.04e-3  # Medio 2, es Hueso parietal de 0.00704 m
L = L1 + L2  # Longitud total

Nx = 1000 #Número de puntos (nodos) espaciales
x = np.linspace(0, L, Nx) #arreglo con la cantidad de nodos
dx = x[1] - x[0] #Delta x, todos tendrán el mismo valor 


#Variables del dominio temporal
#Determinar el valor para tener estabilidad, Courant 
c_max = max(c1, c2)  # Buscamos la velocidad más rápida del sistema
Courant = 0.5   # Factor de seguridad (debe ser <= 1)
dt = Courant * dx / c_max  # Tamaño del paso de tiempo calculado para ser estable

T = 2e-5  # Tiempo total de simulación
Nt = int(round(T / dt)) + 1   # Cantidad total de pasos en el bucle temporal, el +1 es para el tiempo inicial
t = np.linspace(0, Nt*dt, Nt) # Creación del arreglo de tiempo discreto


########## Creación del medio heterogéneo (malla) #################
#Arreglos de ceros
c = np.zeros(Nx) #para guardar la velocidad en cada nodo
alpha = np.zeros(Nx) #guardar la atenuación en cada nodo

#Definir el punto exacto de la interfaz entre tejidos
indice_interfaz = int(round((L1 / L) * Nx)) 
x_interfaz = L1 #Posición de la interfaz en el espacio
#Definir las velocidades en cada medio
c[:indice_interfaz] = c1 #Los nodos donde la velocidad es del medio 1 lado izquierdo
c[indice_interfaz:] = c2 #Los nodos donde la velocidad es del medio 2 lado derecho

########## Calcular atenuación ###########################
# Definir la atenuación para el medio 2 
#Valor maximo que puede tomar alpha en el ultimo nodo
R=1e-6 #valor del coeficiente de refexión antes era 0.001 pero la onda se reflejaba en x=L
alpha_max = np.log(1/R) *((3*c2 )/(2*(L - x_interfaz))) #calculo de alpha_max a partir del coeficiente de reflexión y la longitud del medio

nodos_medio_2 = Nx - indice_interfaz #calcular la cantidad de nodos tiene el medio 2
perfil_crecimiento = np.linspace(0, 1, nodos_medio_2) ** 2 #la atenuación tiene un crecimiento suave 
alpha[indice_interfaz:] = alpha_max * perfil_crecimiento


############### Definición fuente del pulso ##########
#Calcular S
S = A * np.sin(2 * np.pi * f0 * t) * np.exp(-1 * (t - tc)**2 / (2 * sigma**2))


############### Condiciones iniciales y matrices vacias ###############
#Desplazamiento inicial nulo y velocidad inicial nula
u_previo = np.zeros(Nx) #Desplazamiento en el tiempo n-1
u_actual = np.zeros(Nx) #Desplazamiento en el tiempo n
u_nuevo = np.zeros(Nx) #Desplazamiento en el tiempo n+1
u_historial = np.zeros((Nt, Nx)) #Para guardar el desplazamiento en cada nodo a lo largo del tiempo, filas son el tiempo (Nt) y columnas son el espacio (Nx)

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
    u_historial[n, :] = u_actual[:] #guardar el desplazamiento actual en el historial para graficar después 


######################## Graficar resultados ##########################

#Definir los tiempos específicos para graficar
t1= int(Nt*0.15) #tiempo en donde se deberia estar la onda en el medio 1
t2= int(Nt*0.30) #tiempo en donde deberia estar la onda recien transmitiendose pasando por la interfaz y reflejandose
t3= int(Nt*0.38) #tiempo en donde deberia verse la onda ya atenuada y reflejada separadas
t4= int(Nt*0.50) #la onda ya se encuentra atenuada por completo y se ve la reflejada
t5= int(Nt*0.70) #la onda ya se encuentra atenuada por completo y puede que solamente exista la onda que se refleja en x=0

#Graficar por separado, se cambia de manera manual los tiempos por el costo computacional.
#plt.figure(figsize=(7, 5), layout="constrained")
#plt.plot(x * 1000, u_historial[t5, :] * 1e9, color='purple')
#plt.axvline(x=x[indice_interfaz] * 1000, color='k', linestyle='--', alpha=0.5)
#plt.title(f't={t5*dt*1e6:.2f}µs')
#plt.xlabel('Posición x (mm)')
#plt.ylabel('Desplazamiento u (nm)')
#plt.ylim(-0.6, 0.6)
#plt.grid(True)
#plt.show()

## Grafica con numeros mas grandes ##
#Graficar por separado, se cambia de manera manual los tiempos por el costo computacional.
plt.figure(figsize=(7, 5), layout="constrained")
plt.plot(x * 1000, u_historial[t1, :] * 1e9, color='purple')
plt.axvline(x=x[indice_interfaz] * 1000, color='k', linestyle='--', alpha=0.5)

# 1. Agrandar el título y las etiquetas de los ejes
plt.title(f't={t1*dt*1e6:.2f}µs', fontsize=18)
plt.xlabel('Posición x (mm)', fontsize=20)
plt.ylabel('Desplazamiento u (nm)', fontsize=20)

# 2. Agrandar los números (ticks) de los ejes X e Y
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.ylim(-0.6, 0.6)
plt.grid(True)
plt.show() #sacar comentario para ver la grafica, se deja comentado para ver la simulación

############### Simulación #####################################

fig_ani, ax_ani = plt.subplots(figsize=(8, 5), layout="constrained")

# Configuración visual de la ventana de animación
ax_ani.set_xlim(0, L * 1000) 
ax_ani.set_ylim(-1.5, 1.5)
ax_ani.set_xlabel('Posición x (mm)')
ax_ani.set_ylabel('Desplazamiento u(x,t) (nm)')
ax_ani.grid(True)
ax_ani.axvline(x=x_interfaz * 1000, color='k', linestyle='--', alpha=0.5) 

# Crear un objeto de línea vacío 
linea, = ax_ani.plot([], [], color='purple', lw=1.5)

# Función de inicialización
def init():
    linea.set_data([], [])
    return linea,

# Función que actualiza la gráfica en cada cuadro (frame)
salto_temporal = int(Nt / 200) 

def animar(i):
    n = i * salto_temporal
    if n < Nt:
        linea.set_data(x * 1000, u_historial[n, :] * 1e9)
        ax_ani.set_title(f'Tiempo: {n * dt * 1e6:.2f} µs')
    return linea,

# Generar la animación
ani = animation.FuncAnimation(fig_ani, animar, init_func=init, frames=200, interval=30, blit=True)

# Para guardar como GIF 
ani.save('simulacion_ultrasonido.gif', writer='pillow', fps=30, dpi=150)

plt.show()


