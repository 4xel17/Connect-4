# SaraFirstVisitMc — Connect-4 Agent

## Descripción

Este agente implementa una estrategia basada en **Monte Carlo First Visit** para jugar Connect-4.

El agente combina:

- detección de victoria inmediata,
- bloqueo de amenazas del oponente,
- simulaciones aleatorias (rollouts),
- aprendizaje mediante una Q-table,
- exploración epsilon-greedy.

---

# Estructura

El agente principal se encuentra en:

```text
SaraFirstVisitMc.py
```

La clase principal es:
```python
class SaraFirstVisitMc(Policy)
```

# Cómo funciona

El agente toma decisiones siguiendo este orden:

1. Buscar una jugada ganadora inmediata.
2. Bloquear una posible victoria del rival.
3. Evaluar movimientos usando:
4. Monte Carlo rollouts,
5. valores almacenados en la Q-table,
6. bonus estratégico por cercanía al centro.
7. Aplicar exploración aleatoria con epsilon-greedy.

# Características principales
- Monte Carlo First Visit
 El agente actualiza los valores Q únicamente en la primera visita de cada par:

```text
(estado, acción)
```
durante un episodio.

- Rollouts aleatorios
  
Para evaluar movimientos, el agente simula partidas completas usando movimientos aleatorios.


- Q-Table
El agente almacena experiencia previa usando:

```python
q_table[(estado, accion)] = valor
```

- Exploración epsilon-greedy
Con una probabilidad del 10%, el agente explora movimientos aleatorios para evitar sobreajuste.

# Uso

Ejemplo básico:

```python
agent = SaraFirstVisitMc()
agent.mount()

action = agent.act(board)
```

# Dependencias

Instalar NumPy:
```bash
pip install numpy
```

# Notebook de análisis

El archivo:
```text
entrega.ipynb
```

incluye:

- pruebas contra jugador aleatorio,
- auto-juego,
- análisis de rendimiento,
- gráficas y experimentos.
