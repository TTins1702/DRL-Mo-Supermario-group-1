1. #  Tổng quan:

* [https\://mo-gymnasium.farama.org/environments/mo-supermario/](https://mo-gymnasium.farama.org/environments/mo-supermario/)  
* Môi trường:

| Action Space | Discrete(256) |
| :---- | :---- |
| Observation Shape | (240, 256, 3\) |
| Observation High | 255 |
| Observation Low | 0 |
| Reward Shape | (5,) |
| Reward High | \[ inf 0\. 0\. 100\. inf\] |
| Reward Low | \[-inf \-inf \-25. 0\. 0.\] |
| Import | mo\_gymnasium.make("mo-supermario-v0") |

* Description: Multi-objective version of the SuperMarioBro environment.  
* Github: [https\://github.com/Kautenja/gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)  
* **Reward Space**:  
  * 0: How far Mario moved in the x position  
  * 1: Time penalty for how much time has passed between two time steps  
  * 2: \-25 if Mario died, 0 otherwise  
  * 3: \+100 if Mario collected coins, else 0  
  * 4: Points for killing an enemy  
* Episode Termination: The episode terminates when Mario dies or reaches the flag.

2. # Mục tiêu:

   * Vận dụng các kỹ năng đã học trong môn Học tăng cường (Reinforcement Learning) để giải bài toán Mario thỏa mãn các điều kiện sau:  
     * Sử dụng Deep Reinforcement Learning để giải bài toán.  
   * Mario (agent) phải đi, vượt qua các chướng ngại vật tới đích (reaches the flag) màn 1-1 sẽ đạt 8 điểm cuối kỳ.  
   * 2 điểm còn lại (giải quyết sau) sẽ do thầy quy định sau: có thể là thời gian ngắn, điểm số cao, vượt qua nhiều màn,...  
* Tất cả mọi người phải hiểu bài toán, hiểu code, công thức,... Vì thấy sẽ vấn đáp ngẫu nhiên từng người.