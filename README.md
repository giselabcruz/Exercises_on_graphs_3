# **Shortest Path Algorithms Benchmark**

## **Overview**
This project benchmarks two classic shortest-path algorithms — **Breadth-First Search (BFS)** and **Dijkstra’s algorithm** — to compare their **runtime performance** and **scalability** on randomly generated graphs.  
Graphs were generated using an Erdős–Rényi model with different sizes, and each algorithm was executed multiple times to obtain stable average results.

---

## **Experiment Setup**
- **Graph model:** Erdős–Rényi `G(n, p)`  
- **Edge probability (p):** 0.01  
- **Number of trials:** 300  
- **Edge weights (for Dijkstra):** Uniformly distributed in [1, 10]  
- **Graph sizes tested:** 200, 500, 1000, 2000, 5000 nodes  

For every configuration:
1. **BFS** was executed on the **unweighted version** of the graph.  
2. **Dijkstra** was executed on the **weighted version** of the same graph.  
3. Both **runtime** and **internal operations** (queue pushes, edge checks, etc.) were measured.

---

## **Results Summary**

### **1. Runtime Comparison — `runtime_bfs_vs_dijkstra.png`**
- BFS remains **consistently fast**, showing near-zero execution times even for graphs with 5,000 nodes.  
- Dijkstra’s runtime increases much faster as the graph grows, mainly due to **priority queue management** and **distance relaxations**.  
- Overall, BFS is **more scalable** and ideal for **unweighted or sparse graphs**, while Dijkstra is only necessary when edge weights are relevant.

---

### **2. Queue and Priority Queue Operations — `ops_queue_vs_pq.png`**
- BFS shows **linear growth** in queue insertions, matching its O(V + E) complexity.  
- Dijkstra performs **many more push and pop operations** on its priority queue since every distance update can lead to new insertions.  
- The gap between **pushes** and **pops** highlights the **extra workload** of maintaining a heap structure, which directly affects execution time.

---

### **3. Edge Examinations — `ops_edge_exams.png`**
- The number of **edges examined** increases with graph size in both algorithms, but Dijkstra checks **far more edges** than BFS.  
- This happens because Dijkstra often revisits edges during distance relaxation, while BFS simply explores each edge once.  
- As a result, Dijkstra’s computational cost grows faster even with sparse graphs.

---

## **Key Takeaways**
- **BFS** is ideal for **unweighted graphs**, offering simple implementation, linear complexity, and excellent scalability.  
- **Dijkstra** works for **weighted graphs**, but incurs extra cost from maintaining a priority queue and relaxing edges.  
- The **overhead becomes significant** once the graph exceeds a few thousand nodes, even when connectivity is low (`p = 0.01`).  
- The **crossover point** where Dijkstra becomes slower than BFS occurs around **n ≈ 200** nodes.

---

## **📈 Results Visualization**

The following plots summarize the experimental results.  
All generated plots are stored in the `plots/` directory.

### **Average Runtime**
Comparison of the average execution time as the graph grows.  
![Runtime BFS vs Dijkstra](plots/runtime_bfs_vs_dijkstra.png)

---

### **Queue / Priority Queue Operations**
Number of operations performed on regular queues (BFS) and priority queues (Dijkstra).  
![Queue vs Priority Queue Operations](plots/ops_queue_vs_pq.png)

---

### **Edge Examinations**
Average number of edges explored per graph size.  
![Edge Examination Operations](plots/ops_edge_exams.png)

---

## **Conclusion**
In summary:
- **BFS** maintains stable, near-linear performance across all graph sizes, making it highly efficient for unweighted networks.  
- **Dijkstra’s algorithm**, while more flexible for weighted graphs, becomes **computationally heavier** as the number of nodes and edges grows.  
- The experiment clearly shows the trade-off between **speed and generality**:  
  - **BFS** is optimal for fast traversal.  
  - **Dijkstra** is the right choice when **edge weights** must be considered.
