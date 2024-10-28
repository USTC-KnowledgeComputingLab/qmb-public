import logging
import typing
import dataclasses
import torch
import tyro
from .lobpcg import lobpcg
from .common import CommonConfig
from .subcommand_dict import subcommand_dict


@dataclasses.dataclass
class IterConfig:
    common: typing.Annotated[CommonConfig, tyro.conf.OmitArgPrefixes]

    # sampling count
    sampling_count: typing.Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1024

    # selected extended sampling count
    selected_sampling_count: typing.Annotated[int, tyro.conf.arg(aliases=["-s"])] = 65536

    def main(self):
        model, network = self.common.main()

        logging.info(
            "sampling count: %d, selected sampling count: %d",
            self.sampling_count,
            self.selected_sampling_count,
        )

        logging.info("first sampling core configurations")
        configs_core, psi_core, _, _ = network.generate_unique(self.sampling_count)
        logging.info("core configurations sampled")

        while True:
            sampling_count_core = len(configs_core)
            logging.info("core configurations count is %d", sampling_count_core)

            logging.info("calculating extended configurations")
            indices_i_and_j, values, configs_extended = model.outside(configs_core.cpu())
            indices_i_and_j = torch.as_tensor(indices_i_and_j).cuda()
            values = torch.as_tensor(values).cuda()
            configs_extended = torch.as_tensor(configs_extended).cuda()
            logging.info("extended configurations created")
            sampling_count_extended = len(configs_extended)
            logging.info("extended configurations count is %d", sampling_count_extended)

            logging.info("converting sparse extending matrix data to sparse matrix")
            hamiltonian = torch.sparse_coo_tensor(indices_i_and_j.T, values, [sampling_count_core, sampling_count_extended], dtype=torch.complex128).to_sparse_csr()
            logging.info("sparse extending matrix created")

            logging.info("estimating the importance of extended configurations")
            importance = (psi_core.conj() * psi_core).abs() @ (hamiltonian.conj() * hamiltonian).abs()
            importance[:sampling_count_core] += importance.max()
            logging.info("importance of extended configurations created")

            logging.info("selecting extended configurations by importance")
            selected_indices = importance.sort(descending=True).indices[:self.selected_sampling_count].sort().values
            logging.info("extended configurations selected indices prepared")

            logging.info("selecting extended configurations")
            configs_extended = configs_extended[selected_indices]
            logging.info("extended configurations selected")
            sampling_count_extended = len(configs_extended)
            logging.info("selected extended configurations count is %d", sampling_count_extended)

            logging.info("calculating sparse data of hamiltonian on extended configurations")
            indices_i_and_j, values = model.inside(configs_extended.cpu())
            indices_i_and_j = torch.as_tensor(indices_i_and_j).cuda()
            values = torch.as_tensor(values).cuda()
            logging.info("converting sparse matrix data to sparse matrix")
            hamiltonian = torch.sparse_coo_tensor(indices_i_and_j.T, values, [sampling_count_extended, sampling_count_extended], dtype=torch.complex128).to_sparse_csr()
            logging.info("sparse matrix on extended configurations created")

            logging.info("preparing initial psi used in lobpcg")
            psi_extended = torch.cat([psi_core, torch.zeros([sampling_count_extended - sampling_count_core], dtype=psi_core.dtype, device=psi_core.device)], dim=0).view([-1, 1])
            logging.info("initial psi used in lobpcg has been created")

            logging.info("calculating minimum energy on extended configurations")
            energy, psi_extended = lobpcg(hamiltonian, psi_extended, maxiter=1024)
            logging.info("energy on extended configurations is %.10f, ref energy is %.10f, error is %.10f", energy.item(), model.ref_energy, energy.item() - model.ref_energy)

            logging.info("calculating indices of new core configurations")
            indices = torch.argsort(psi_extended.abs().flatten())[-self.sampling_count:]
            logging.info("indices of new core configurations has been obtained")

            logging.info("update new core configurations")
            configs_core = configs_extended[indices]
            psi_core = psi_extended[indices].flatten()
            logging.info("new core configurations has been updated")


subcommand_dict["iter"] = IterConfig
