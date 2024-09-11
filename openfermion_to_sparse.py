def openfermion_to_sparse(
    openfermion_hamiltonian: list[tuple[tuple[tuple[int, int], ...], complex]],
    configs: list[tuple[int, ...]],
) -> tuple[list[int], list[int], list[complex]]:
    config_dict: dict[tuple[int, ...], int] = {config: index for index, config in enumerate(configs)}

    indices_i: list[int] = []
    indices_j: list[int] = []
    values: list[complex] = []
    for index_i, config_i in enumerate(configs):
        for key, value in openfermion_hamiltonian:
            config_j: list[int] = list(config_i)
            success: bool = True
            for site, operation in reversed(key):
                if operation == 0:
                    if config_j[site] != 1:
                        success = False
                        break
                    config_j[site] = 0
                    if sum(config_j[:site]) % 2 == 1:
                        value = -value
                else:
                    if config_j[site] != 0:
                        success = False
                        break
                    config_j[site] = 1
                    if sum(config_j[:site]) % 2 == 1:
                        value = -value
            if success:
                index_j = config_dict.get(tuple(config_j), -1)
                if index_j != -1:
                    indices_i.append(index_i)
                    indices_j.append(index_j)
                    values.append(value)

    return indices_i, indices_j, values
