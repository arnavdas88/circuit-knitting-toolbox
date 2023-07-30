import numpy as np

from qiskit import QuantumCircuit, Aer
from qiskit.providers.fake_provider import FakeManila, FakeNairobi, FakeHanoi

from qiskit_aer.noise import NoiseModel

from qiskit_ibm_runtime import QiskitRuntimeService, Options
from circuit_knitting.cutting.cutqc.wire_cutting import cut_circuit_wires, reconstruct_full_distribution, evaluate_subcircuits
from circuit_knitting.cutting.cutqc.wire_cutting_verification import verify

num_qubits = 5

circuit = QuantumCircuit(num_qubits)
for i in range(num_qubits):
    circuit.h(i)
circuit.cx(0, 1)
for i in range(2, num_qubits):
    circuit.t(i)
circuit.cx(0, 2)
circuit.rx(np.pi / 2, 4)
circuit.rx(np.pi / 2, 0)
circuit.rx(np.pi / 2, 1)
circuit.cx(2, 4)
circuit.t(0)
circuit.t(1)
circuit.cx(2, 3)
circuit.ry(np.pi / 2, 4)
for i in range(num_qubits):
    circuit.h(i)

# Use local versions of the primitives by default.
service = QiskitRuntimeService(channel="ibm_quantum", instance='ibm-q/open/main', token="39b90792f9a58b18fe1d8ab6619a3078a88d801b32fe136b71cdadeb51e5db127b8796ab3c19e67ab8359bbc552d647a153f5bb12d62930ed80193aa0437ec43")

# Make a noise model
fake_backend = FakeHanoi()
noise_model = NoiseModel.from_backend(fake_backend)

# Set the Sampler and runtime options
options = Options(
            execution = {"shots": 4000},
            simulator = {
                "noise_model": noise_model,
                "basis_gates": fake_backend.configuration().basis_gates,
                "coupling_map": fake_backend.configuration().coupling_map,
                "seed_simulator": 42,
                "max_parallel_threads": 20,
                "max_parallel_shots": 512,
                "max_parallel_experiments": 20,
                "precision": "single",
            },
            optimization_level = 3,
            resilience_level = 1
        )

cuts = cut_circuit_wires(
    circuit=circuit, method="manual", subcircuit_vertices=[[0, 1], [2, 3]]
)

if __name__ == "__main__":

    # subcircuit_instance_probabilities = evaluate_subcircuits(cuts)

    # Uncomment the following lines to instead use Qiskit Runtime Service as configured above.
    # subcircuit_instance_probabilities = evaluate_subcircuits(cuts,
    #                                                          service = service,
    #                                                          backend_names = ["ibmq_qasm_simulator"],
    #                                                          options = options,
    #                                                         )

    # subcircuit_instance_probabilities = evaluate_subcircuits(cuts,
    #                                                          service = None,
    #                                                          backend_names = ["aer_simulator_density_matrix"],
    #                                                          options = options,
    #                                                         )
    subcircuit_instance_probabilities = evaluate_subcircuits(cuts,
                                                             service = None,
                                                             backend_names = [Aer.get_backend("aer_simulator_density_matrix")],
                                                             options = Options(),
                                                            )

    reconstructed_probabilities = reconstruct_full_distribution(
        circuit, subcircuit_instance_probabilities, cuts
    )


    metrics, exact_probabilities = verify(circuit, reconstructed_probabilities)

    print(metrics)
    print(reconstructed_probabilities)