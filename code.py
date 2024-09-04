import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


n = 101  # 时间切片数量
grid_size = 256 #网格尺寸
dt = 30 / (n - 1)  # 时间间隔
nu = 4.66e-4  # 粘性系数
epochs = 100  # 训练次数
lr = 0.01  # 学习率


def fft(omega):
    return torch.fft.fft2(omega)


def ifft(F_omega):
    return torch.fft.ifft2(F_omega).real


class Fluid_Solver:

    def __init__(self):
        super(Fluid_Solver, self).__init__()
        self.kx = torch.fft.fftfreq(grid_size).reshape(-1, 1) * 2 * np.pi#波数空间内的x坐标
        self.ky = torch.fft.fftfreq(grid_size).reshape(1, -1) * 2 * np.pi#波束空间内的y坐标
        self.k2 = self.kx**2 + self.ky**2
        self.k2[0, 0] = 1
        self.F_w = torch.rand(n, grid_size, grid_size, dtype=torch.cfloat, requires_grad=True)

    def compute_time_derivative(self, Delta_t):#∂F(ω)/∂t
        F_w_dt = torch.zeros_like(self.F_w)
        F_w_dt[:-1] = (self.F_w[1:] - self.F_w[:-1]) / Delta_t
        F_w_dt[-1] = (self.F_w[-1] - self.F_w[-2]) / Delta_t
        return F_w_dt

    def compute_variants(self):
        F_vx = -self.ky / self.k2 * self.F_w
        F_vy = self.kx / self.k2 * self.F_w
        F_lap_w = -self.k2 * self.F_w#F(∇²ω)
        F_advection = fft(ifft(self.F_w) * ifft(F_vx)) * self.kx + fft(
            ifft(self.F_w) * ifft(F_vy)) * self.ky#F(v*∇ω)
        F_w_dt = self.compute_time_derivative(dt)
        return F_vx, F_vy, F_lap_w, F_advection, F_w_dt

    def compute_physics(self, F_vx, F_vy):
        vx = ifft(F_vx).real
        vy = ifft(F_vy).real
        return vx, vy


def compute_loss(F_lap_w, F_advection, F_w_dt, nu):
    #损失函数
    loss = torch.mean(torch.pow(torch.abs(F_advection - nu * F_lap_w + F_w_dt),2))
    return loss


solver = Fluid_Solver()
optimizer = torch.optim.Adam([solver.F_w], lr=lr)#设置优化器
loss_history = []

for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
    #开始训练
    F_vx, F_vy, F_lap_w, F_advection, F_w_dt = solver.compute_variants()

    optimizer.zero_grad()
    loss = compute_loss(F_advection, F_lap_w, F_w_dt, nu)
    loss.backward()#计算梯度
    optimizer.step()#优化F(ω)

    if epoch % 5 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        loss_history.append(loss.item())

#训练完成后测试
F_vx, F_vy, F_lap_w, F_advection, F_w_dt = solver.compute_variants()
vx, vy = solver.compute_physics(F_vx, F_vy)
vx = vx.detach().numpy()
vy = vy.detach().numpy()
test_loss = torch.mean(torch.abs(ifft(F_advection - nu * F_lap_w + F_w_dt)))
print(test_loss)

#绘制误差曲线
plt.figure()
plt.plot(loss_history)
plt.savefig('./train_loss.png')

#可视化
fig, ax = plt.subplots()
X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
quiver = ax.quiver(X, Y, vx[0], vy[0])
ax.set_xlim(0, grid_size - 1)
ax.set_ylim(0, grid_size - 1)
ax.set_title("Fluid Flow Evolution")


def update(frame_number):
    quiver.set_UVC(vx[frame_number], vy[frame_number])
    ax.set_title(f"Fluid Flow at Time Slice {frame_number}")
    return quiver, ax


ani = FuncAnimation(fig, update, frames=n, interval=50, blit=True)

plt.show()

