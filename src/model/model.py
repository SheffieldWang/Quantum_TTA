
                 # 导入numpy库并简写为np
from mindquantum.core.circuit import Circuit                # 导入Circuit模块，用于搭建量子线路
from mindquantum.core.gates import  RY, RZ ,X          # 导入量子门H, RX, RY, RZ
from mindquantum.core.parameterresolver import PRGenerator  # 导入参数生成器模块
from mindquantum.core.operators import QubitOperator           # 导入QubitOperator模块，用于构造泡利算符
from mindquantum.core.operators import Hamiltonian             # 导入Hamiltonian模块，用于构建哈密顿量
from mindquantum.simulator import Simulator
from mindquantum.algorithm.library import amplitude_encoder

class QuantumCircuit:
    def __init__(self, n_qubits, n_layers):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.simulator = Simulator('mqvector', n_qubits)
        self.observable =  Hamiltonian(QubitOperator('Z0', 1))   
        
        

    def RyEncoder(self):
        alpha = PRGenerator('alpha')
        encoder = Circuit()                   # 初始化量子线路
        for i in range(self.n_qubits):                    # 4量子比特
            encoder += RY(alpha.new()).on(i)  # RY门作用在第i位量子比特
        encoder = encoder.no_grad()           # Encoder作为整个量子神经网络的第一层，不用对编码线路中的梯度求导数，因此加入no_grad()
        encoder.as_encoder()  
        return encoder
        
    def ansatz(self):
        theta = PRGenerator('theta')                 # 初始化参数生成器
        ansatz = Circuit()                           # 初始化量子线路                # 设置层数
        for l in range(self.n_layers):                    # l层循环
            for i in range(self.n_qubits):                # 对每个量子比特应用旋转门
                ansatz += RZ(theta.new()).on(i)      # RZ门作用在第i位量子比特
                ansatz += RY(theta.new()).on(i)      # RY门作用在第i位量子比特
                ansatz += RZ(theta.new()).on(i)      # RZ门作用在第i位量子比特
            for i in range(self.n_qubits):                # 顺序连接的CNOT门
                ansatz += X.on((i+1)%self.n_qubits, i)    # CNOT门，控制位为i，目标位为i+1
        ansatz.as_ansatz()  
        return ansatz
        
    
    def get_grad_ops(self):
        encoder = self.RyEncoder()
        ansatz = self.ansatz()
        circuit = encoder.as_encoder() + ansatz.as_ansatz()
        grad_ops = self.simulator.get_expectation_with_grad(self.observable,
                                         circuit)
        return grad_ops
