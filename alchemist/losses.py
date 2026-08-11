import torch
from alchemist.neighbors import compute_neighbor_masks, get_neighbor_indices, assemble_neighbor_features
from alchemist.analysis import compute_energy_parameterized
from alchemist.pdbs import read_pdb_coords

class KLLoss:
    def __init__(self, pdb_path, box, cuts, MU, E1, E2, KT):
        coords = read_pdb_coords(pdb_path)
        box = torch.tensor(box, dtype=torch.float32)
        masks = compute_neighbor_masks(coords, box, cuts)
        self.neighborlists = get_neighbor_indices(masks)
        self.MU = torch.tensor(MU) if not isinstance(MU, torch.Tensor) else MU
        self.E1 = torch.tensor(E1) if not isinstance(E1, torch.Tensor) else E1
        self.E2 = torch.tensor(E2) if not isinstance(E2, torch.Tensor) else E2
        self.KT = KT

    def __call__(self, predictions, batch):
        # predictions is (x, logJ, info) from MultiStep.forward
        x, logJ, info = predictions
        x0 = batch
        device = x['r'].device
        
        # Ensure parameters are on the correct device
        mu = self.MU.to(device) if isinstance(self.MU, torch.Tensor) else torch.tensor(self.MU, dtype=torch.float32, device=device)
        e1 = self.E1.to(device) if isinstance(self.E1, torch.Tensor) else torch.tensor(self.E1, dtype=torch.float32, device=device)
        e2 = self.E2.to(device) if isinstance(self.E2, torch.Tensor) else torch.tensor(self.E2, dtype=torch.float32, device=device)
        
        # We need to handle neighborlists. Since they are lists of tensors, move each one.
        # In the current implementation, self.neighborlists is not yet a buffer, 
        # but we can compute it or store it. 
        # For now, let's assume we need to compute it or it's stored as a list.
        # Since we are in __call__, we should have them ready.
        
        # Re-computing or moving neighborlists:
        # To avoid re-computing every step, we should have stored them.
        # Let's assume we store them in __init__ as a list of tensors.
        
        # If self.neighborlists is a list of tensors:
        nbrs = [n.to(device) for n in self.neighborlists]
        
        assembled = assemble_neighbor_features(x['r'], nbrs)
        energy = compute_energy_parameterized(assembled, mu, e1, e2)
        
        assembled0 = assemble_neighbor_features(x0['r'], nbrs)
        energy0 = compute_energy_parameterized(assembled0, mu, e1, e2)
        
        loss = 1 / self.KT * (energy.sum(1) - energy0.sum(1)) - logJ
        return loss.mean()
