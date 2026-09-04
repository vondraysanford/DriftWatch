// DriftWatch durable Azure footprint (see CLAUDE.md decision 18).
// Your object ID: az ad signed-in-user show --query id -o tsv
// Preview: az deployment sub what-if --name driftwatch-infra --location eastus2 --template-file infra/main.bicep --parameters alertEmail=<you> storageDataPrincipalId=<your object id>
// Deploy:  az deployment sub create  --name driftwatch-infra --location eastus2 --template-file infra/main.bicep --parameters alertEmail=<you> storageDataPrincipalId=<your object id>
targetScope = 'subscription'

@description('Region for the resource group and all resources')
param location string = 'eastus2'

@description('Monthly budget alert amount in USD')
param budgetAmount int = 30

@description('First day of the budget period. Azure cannot update it after creation; only override after a full teardown (first of the current month).')
param budgetStartDate string = '2026-08-01'

@description('Recipient for budget alerts. Deploy-time parameter only; never commit a value.')
param alertEmail string

@description('Object ID of the developer identity granted blob data access for DVC push/pull. Deploy-time parameter only; never commit a value.')
param storageDataPrincipalId string

resource rg 'Microsoft.Resources/resourceGroups@2024-03-01' = {
  name: 'DriftWatch'
  location: location
}

module workspace 'modules/workspace.bicep' = {
  name: 'driftwatch-workspace'
  scope: rg
  params: {
    location: location
    storageDataPrincipalId: storageDataPrincipalId
  }
}

// Durable Container Apps footprint. The app itself is deployed separately by CI
// (infra/containerapp.bicep) because its image tag changes on every merge.
module containerAppEnv 'modules/containerapp-env.bicep' = {
  name: 'driftwatch-containerapp-env'
  scope: rg
  params: {
    location: location
    logAnalyticsName: workspace.outputs.logAnalyticsName
    acrName: workspace.outputs.acrName
    storageAccountName: workspace.outputs.storageAccountName
  }
}

module budget 'modules/budget.bicep' = {
  name: 'driftwatch-budget'
  scope: rg
  params: {
    budgetAmount: budgetAmount
    startDate: budgetStartDate
    alertEmail: alertEmail
  }
}

output workspaceName string = workspace.outputs.workspaceName
output acrName string = workspace.outputs.acrName
output acrLoginServer string = workspace.outputs.acrLoginServer
output storageAccountName string = workspace.outputs.storageAccountName
output containerAppEnvironment string = containerAppEnv.outputs.environmentName
output containerAppIdentityClientId string = containerAppEnv.outputs.identityClientId
