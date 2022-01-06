
# Environment Variables

System specific variables (e.g. absolute paths to datasets) should not be under version control, otherwise there will be conflicts between different users.

The best way to handle system specific variables is through environment variables.

You can define new environment variables in a `.env` file in the project root. A copy of this file (e.g. `.env.template`) can be under version control to ease new project configurations.

To define a new variable write inside `.env`:

```bash
export MY_VAR=/home/user/my_system_path
```

You can dynamically resolve the variable name from Python code with:

```python
get_env("MY_VAR")
```

and in the Hydra `.yaml` configuration files with:

```yaml
${oc.env:MY_VAR}
```
