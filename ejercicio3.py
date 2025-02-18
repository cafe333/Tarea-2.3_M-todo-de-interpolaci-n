import numpy as np
import matplotlib.pyplot as plt

# Función original
def f(x):
    return np.exp(-x) - x

# Interpolación de Lagrange
def lagrange_interpolation(x, x_points, y_points):
    n = len(x_points)
    result = 0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

# Método de Bisección con impresión de errores y almacenamiento de iteraciones
def bisect(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) > 0:
        raise ValueError("El intervalo no contiene una raíz")
    
    iter_count = 0
    prev_c = a
    iteraciones = []
    
    print("Iteración | x | Error absoluto | Error relativo | Error cuadrático")
    for _ in range(max_iter):
        c = (a + b) / 2
        error_abs = abs(c - prev_c)
        error_rel = abs(error_abs / c) if c != 0 else 0
        error_cuad = error_abs ** 2
        print(f"{iter_count + 1:9d} | {c:.6f} | {error_abs:.6e} | {error_rel:.6e} | {error_cuad:.6e}")
        
        iteraciones.append((iter_count + 1, c))  # Guardar iteración y valor de x
        
        if abs(func(c)) < tol or (b - a) / 2 < tol:
            return c, iteraciones
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c
        prev_c = c
        iter_count += 1

    return (a + b) / 2, iteraciones  # Retorna la mejor estimación de la raíz y las iteraciones

# Selección de cuatro puntos de interpolación en el intervalo [0,1]
x0 = 0.0
x1 = 0.3
x2 = 0.6
x3 = 1.0
x_points = np.array([x0, x1, x2, x3])
y_points = f(x_points)

# Construcción del polinomio interpolante
x_vals = np.linspace(x0, x3, 100)
y_interp = [lagrange_interpolation(x, x_points, y_points) for x in x_vals]

# Encontrar raíz del polinomio interpolante usando bisección
target_root, iteraciones = bisect(lambda x: lagrange_interpolation(x, x_points, y_points), x0, x3)

# Gráfica 1: Función original e interpolación
plt.figure(figsize=(8, 6))
plt.plot(x_vals, f(x_vals), label="f(x) = e^(-x) - x", linestyle='dashed', color='blue')
plt.plot(x_vals, y_interp, label="Interpolación de Lagrange", color='red')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(target_root, color='green', linestyle='dotted', label=f"Raíz aproximada: {target_root:.6f}")
plt.scatter(x_points, y_points, color='black', label="Puntos de interpolación")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Interpolación y búsqueda de raíces")
plt.legend()
plt.grid(True)
plt.savefig("interpolacion_raices.png")  # Guarda la imagen
plt.show()

# Gráfica 2: Convergencia de la raíz en el método de bisección
plt.figure(figsize=(8, 6))
iter_nums, iter_xs = zip(*iteraciones)
plt.plot(iter_nums, iter_xs, marker='o', linestyle='-', color='purple', label="Aproximaciones de la raíz")
plt.axhline(target_root, color='green', linestyle='dotted', label=f"Raíz aproximada: {target_root:.6f}")
plt.xlabel("Iteración")
plt.ylabel("Valor de x")
plt.title("Convergencia de la raíz con Bisección")
plt.legend()
plt.grid(True)
plt.savefig("convergencia_biseccion.png")  # Guarda la imagen
plt.show()

# Imprimir la raíz encontrada
print(f"La raíz aproximada usando interpolación es: {target_root:.6f}")
