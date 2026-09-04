# OIDC setup for GitHub Actions

Deliberately outside Bicep (decision 18): Entra app registrations are directory objects, not
resource-group resources, and ARM cannot create them. These are the exact commands that were run,
recorded here so the identity can be rebuilt or audited without guesswork.

**Nothing here is a secret.** No client secret is ever created. GitHub proves who it is with a
short-lived token it signs itself, and Azure trusts that token only when it names this repository
and this branch. There is nothing to leak and nothing to rotate.

## 1. App registration and service principal

```bash
APP_ID=$(az ad app create \
  --display-name driftwatch-github-oidc \
  --sign-in-audience AzureADMyOrg \
  --query appId -o tsv)

az ad sp create --id "$APP_ID"
APP_OBJECT_ID=$(az ad app show --id "$APP_ID" --query id -o tsv)
```

## 2. Federated credential, scoped to one repository and one branch

The `subject` is the whole security boundary. A workflow running on any other branch, or in any
other repository, presents a different subject and is refused.

```bash
az ad app federated-credential create --id "$APP_OBJECT_ID" --parameters '{
  "name": "github-main",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:vondraysanford/DriftWatch:ref:refs/heads/main",
  "description": "GitHub Actions on the main branch of vondraysanford/DriftWatch",
  "audiences": ["api://AzureADTokenExchange"]
}'
```

Verify no password credential exists (the expected answer is `0`):

```bash
az ad app credential list --id "$APP_OBJECT_ID" --query "length(@)" -o tsv
```

### The subject GitHub actually presents

The first run failed with `AADSTS700213: No matching federated identity record found for
presented assertion subject`. GitHub issued an **ID-qualified** subject, not the documented
name-based one:

```
presented:  repo:vondraysanford@101304529/DriftWatch@1333783711:ref:refs/heads/main
configured: repo:vondraysanford/DriftWatch:ref:refs/heads/main
```

The numbers are the account ID and the repository ID (confirmed against
`api.github.com/users/vondraysanford` and `.../repos/vondraysanford/DriftWatch`). This form is
stronger than the name-based one: renaming the repository or the account does not move the trust,
and nobody who later claims the freed-up name can authenticate, because their repository ID differs.

The fix is to match what is presented. Read the failing run's error rather than assuming a format:

```bash
az ad app federated-credential create --id "$APP_OBJECT_ID" --parameters '{
  "name": "github-main-ids",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:<owner>@<owner-id>/<repo>@<repo-id>:ref:refs/heads/main",
  "audiences": ["api://AzureADTokenExchange"]
}'
```

Both credentials are kept. An app may hold several, only one has to match, and keeping the
name-based one costs nothing if GitHub ever issues that form instead.

## 3. RBAC, scoped per resource rather than per subscription

`Contributor` on the resource group covers the control plane (deploying the Container Apps
template). It does **not** grant data-plane access, so pushing images, reading the model registry,
and downloading model artifacts each need their own role.

```bash
SP_OBJECT_ID=$(az ad sp show --id "$APP_ID" --query id -o tsv)
RG="/subscriptions/<subscription-id>/resourceGroups/DriftWatch"

az role assignment create --assignee-object-id "$SP_OBJECT_ID" --assignee-principal-type ServicePrincipal \
  --role "AcrPush" --scope "$RG/providers/Microsoft.ContainerRegistry/registries/<acr-name>"

az role assignment create --assignee-object-id "$SP_OBJECT_ID" --assignee-principal-type ServicePrincipal \
  --role "Contributor" --scope "$RG"

az role assignment create --assignee-object-id "$SP_OBJECT_ID" --assignee-principal-type ServicePrincipal \
  --role "AzureML Data Scientist" --scope "$RG/providers/Microsoft.MachineLearningServices/workspaces/<workspace-name>"

az role assignment create --assignee-object-id "$SP_OBJECT_ID" --assignee-principal-type ServicePrincipal \
  --role "Storage Blob Data Contributor" --scope "$RG/providers/Microsoft.Storage/storageAccounts/<storage-account>"
```

| Role | Scope | Why the pipeline needs it |
|---|---|---|
| AcrPush | container registry | push the serving image |
| Contributor | resource group | deploy `infra/containerapp.bicep` |
| AzureML Data Scientist | ML workspace | read the registered model version |
| Storage Blob Data Contributor | storage account | download model artifacts from workspace storage |

## 4. Repository variables

Set these as GitHub **variables** (Settings, Secrets and variables, Actions, Variables tab), not
secrets. Every one is an identifier that appears in `az` output and in resource IDs.

| Variable | Where it comes from |
|---|---|
| `AZURE_CLIENT_ID` | `az ad app show --id <app> --query appId -o tsv` |
| `AZURE_TENANT_ID` | `az account show --query tenantId -o tsv` |
| `AZURE_SUBSCRIPTION_ID` | `az account show --query id -o tsv` |
| `AZURE_RESOURCE_GROUP` | `DriftWatch` |
| `AZURE_ML_WORKSPACE` | `az deployment sub show -n driftwatch-infra --query properties.outputs.workspaceName.value -o tsv` |
| `ACR_NAME` | ... `properties.outputs.acrName.value` |
| `AZURE_STORAGE_ACCOUNT` | ... `properties.outputs.storageAccountName.value` |
| `DRIFTWATCH_MODEL_NAME` | the registry name from `.env` |

## Teardown

```bash
az ad app delete --id "$APP_ID"   # removes the service principal and the federated credential
```

Role assignments referencing a deleted principal disappear with it.
