// Durable half of the Container Apps footprint: the managed environment, the identity the app
// runs as, and that identity's permissions. None of this changes between deploys.
//
// The app itself lives in infra/containerapp.bicep because its image tag changes on every merge.
// Keeping it out of main.bicep means a later `az deployment sub create` cannot silently revert
// production to a placeholder image.
//
// The identity is user-assigned rather than system-assigned so the role assignments below are made
// once, here, instead of after each app creation; it also survives deleting and recreating the app.

@description('Region, inherited from main')
param location string

@description('Existing Log Analytics workspace that receives container logs')
param logAnalyticsName string

@description('Existing container registry the app pulls images from')
param acrName string

@description('Existing storage account that receives prediction logs')
param storageAccountName string

@description('Blob container for prediction logs')
param predictionContainerName string = 'predictions'

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' existing = {
  name: logAnalyticsName
}

resource acr 'Microsoft.ContainerRegistry/registries@2022-12-01' existing = {
  name: acrName
}

resource storage 'Microsoft.Storage/storageAccounts@2023-01-01' existing = {
  name: storageAccountName
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-driftwatch-app'
  location: location
}

// Pull images from the registry without admin credentials.
var acrPullRoleId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, identity.id, acrPullRoleId)
  scope: acr
  properties: {
    roleDefinitionId: acrPullRoleId
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Write prediction logs as itself: no connection string, no account key (decision 9).
var blobContributorRoleId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')

resource blobWrite 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, identity.id, blobContributorRoleId)
  scope: storage
  properties: {
    roleDefinitionId: blobContributorRoleId
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-driftwatch'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

output environmentId string = environment.id
output environmentName string = environment.name
output identityId string = identity.id
output identityClientId string = identity.properties.clientId
output acrLoginServer string = acr.properties.loginServer
output storageAccountName string = storage.name
output predictionContainerName string = predictionContainerName
