

# **Fundamentos de Inteligencia Artificial – 2026.1**

---

# OhYes – Agente Connect-4 con Online Policy Improvement

## Descripción

`OhYes` es un agente para Connect-4 que implementa **Online Policy Improvement (OPI)**: un proceso de refinamiento de política en tiempo real que, en cada estado del juego, estima qué acción es más prometedora antes de comprometerse con ella.

A diferencia de un agente de búsqueda tradicional, OhYes no construye un árbol global desde la raíz. En cambio, en cada estado del camino principal lanza un mini sub-proceso de exploración local —basado en simulaciones de baja calidad llamadas *trash trials*— para estimar los valores `q(sₜ, a)` de las acciones disponibles. Esas estimaciones guían la decisión en ese estado sin contaminar los valores del proceso principal.

### Idea clave

> En vez de generar simulaciones desde el estado inicial al azar, en cada estado `sₜ` del camino se corre un sub-proceso de exploración que estima qué acción es mejor **desde ahí**. Esto fija buenas decisiones tempranas, haciendo que los estados profundos del árbol sean visitados desde posiciones ya prometedoras.

### Flujo de decisión por turno

1. **Victoria inmediata** – Si existe una jugada ganadora, se juega directamente.
2. **Bloqueo** – Si el oponente tiene una jugada ganadora, se bloquea.
3. **OPI** – Si no hay decisión obvia, se lanza el sub-proceso de mejora de política para estimar la mejor acción desde el estado actual.

Las heurísticas tácticas de los pasos 1 y 2 actúan como "cortocircuito", reservando el presupuesto de simulaciones para situaciones donde la búsqueda realmente aporta valor.

---

## Estructura del proyecto

```
OhYes/
├── policy.py        # Código completo del agente
├── entrega.ipynb    # Análisis empírico y gráficas
└── README.md        # Este archivo
```

---

## Requisitos

```bash
pip install numpy matplotlib pydantic
```

> El módulo `connect4` debe estar disponible en el path (incluido en el repositorio del curso).

---

## Guía de uso rápido

```python
from policy import OhYes

# Crear el agente
agent = OhYes(n_simulations=120)
agent.mount()

# Obtener acción dado un estado del tablero
action = agent.act(board_state)  # board_state: np.ndarray (6×7)
```

### Parámetros

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `n_simulations` | `int` | `120` | Número de *trash trials* por estado en el sub-proceso OPI. Mayor valor = mejor estimación, mayor tiempo de cómputo. |

---

## Rendimiento

Las siguientes métricas muestran la tasa de victoria de OhYes contra un agente aleatorio, variando el número de simulaciones:

| Configuración | vs Aleatorio (Rojo) | vs Aleatorio (Amarillo) |
|---------------|:-------------------:|:-----------------------:|
| `n_sim = 120` | ~86 % | ~82 % |
| `n_sim = 60`  | ~78 % | ~74 % |
| `n_sim = 30`  | ~68 % | ~65 % |

> Jugar como Rojo (primer turno) ofrece ventaja estructural en Connect-4, lo que se refleja en las métricas.

---

## Análisis empírico

El notebook `entrega.ipynb` contiene:

- Comparación OhYes vs Agente Aleatorio (ambos colores)
- Impacto del número de simulaciones en la tasa de victoria
- Auto-juego: OhYes vs OhYes
- Comparación con versión sin heurísticas tácticas (OPI puro)

---

## ¿Por qué Online Policy Improvement?

| Enfoque | Descripción | Limitación |
|--------|-------------|------------|
| Aleatorio puro | Simulaciones sin guía | Desperdicia presupuesto en acciones malas |
| MCTS estándar | Árbol global con UCB1 | Decisiones tempranas débiles hasta convergencia |
| **OPI (OhYes)** | Sub-proceso local por estado | Fija buenas decisiones tempranas; estados profundos visitados desde posiciones prometedoras |

El punto crítico es que los *trash trials* del sub-proceso **no contaminan** los q-values del proceso principal: sirven únicamente para orientar la decisión local y luego se descartan.

---

## Autores

Proyecto desarrollado para el curso **Fundamentos de Inteligencia Artificial – 2026.1**.
