// DriftWatch durable Azure footprint (see CLAUDE.md decision 18).
// Preview: az deployment sub what-if --name driftwatch-infra --location eastus2 --template-file infra/main.bicep --parameters alertEmail=<you>
// Deploy:  az deployment sub create  --name driftwatch-infra --location eastus2 --template-file infra/main.bicep --parameters alertEmail=<you>
targetScope = 'subscription'

@description('Region for the resource group and all resources')
param location string = 'eastus2'

@description('Monthly budget alert amount in USD')
param budgetAmount int = 30

@description('Recipient for budget alerts. Deploy-time parameter only; never commit a value.')
param alertEmail string

resource rg 'Microsoft.Resources/resourceGroups@2024-03-01' = {
  name: 'DriftWatch'
  location: location
}

module workspace 'modules/workspace.bicep' = {
  name: 'driftwatch-workspace'
  scope: rg
  params: {
    location: location
  }
}

module budget 'modules/budget.bicep' = {
  name: 'driftwatch-budget'
  scope: rg
  params: {
    budgetAmount: budgetAmount
    alertEmail: alertEmail
  }
}

output workspaceName string = workspace.outputs.workspaceName
output acrLoginServer string = workspace.outputs.acrLoginServer
output storageAccountName string = workspace.outputs.storageAccountName
