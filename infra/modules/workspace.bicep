// Azure ML workspace with its required companions (storage, key vault, app insights),
// the container registry, and the blob containers later phases depend on
// (dvc = DVC remote, predictions = serving logs).

@description('Region, inherited from main')
param location string

// Stable per resource group, so teardown + redeploy reproduces the same names.
var suffix = uniqueString(resourceGroup().id)

resource storage 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: 'stdriftw${suffix}' // storage names: max 24 chars, lowercase alphanumeric
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  parent: storage
  name: 'default'
}

resource dvcContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: 'dvc'
  properties: {
    publicAccess: 'None'
  }
}

resource predictionsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: 'predictions'
  properties: {
    publicAccess: 'None'
  }
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'kv-driftw-${suffix}' // vault names: max 24 chars
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: true
    // Soft delete reserves the name after deletion; with the deterministic name above,
    // run `az keyvault purge` after a teardown before redeploying.
    softDeleteRetentionInDays: 7
  }
}

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-driftwatch'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-driftwatch'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

resource acr 'Microsoft.ContainerRegistry/registries@2022-12-01' = {
  name: 'acrdriftwatch${suffix}' // registry names: alphanumeric only
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
  }
}

resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: 'mlw-driftwatch'
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: 'DriftWatch'
    storageAccount: storage.id
    keyVault: keyVault.id
    applicationInsights: appInsights.id
    containerRegistry: acr.id
    publicNetworkAccess: 'Enabled'
  }
}

output workspaceName string = mlWorkspace.name
output acrLoginServer string = acr.properties.loginServer
output storageAccountName string = storage.name
