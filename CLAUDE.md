# TabDiffFlow Project Guidelines

## Commands

- Run benchmark example: `python tabdifflow.py --data <csv_file> --target <target_column> --task classification`
- Run TabularDiffFlow example: `python tabular_diffflow.py`
- Generate synthetic data from trained model:
  ```python
  from tabular_diffflow import train_tabular_diffflow_model, generate_synthetic_data
  model = train_tabular_diffflow_model(df, n_epochs=50, batch_size=64)
  synthetic_df = generate_synthetic_data(model[0], n_samples, device, model[1], model[2])
  ```

## Code Style Guidelines

- **Imports**: Group imports (stdlib, third-party, local) with stdlib first, then third-party packages, then local modules
- **Formatting**: Use 4 spaces for indentation, max line length 100 characters
- **Types**: Use type hints for function parameters and return values
- **Naming**:
  - Classes: CamelCase (`TabularDataset`, `TabularDiffFlow`)
  - Functions/methods: snake_case (`train_tabular_diffflow_model`)
  - Variables: snake_case (`feature_ranges`, `batch_size`)
- **Error handling**: Use explicit exception handling with specific exception types
- **Documentation**: Use docstrings for all functions/classes (numpy style with Args and Returns)
- **Numerical code**: Add epsilon values to denominators to prevent division by zero