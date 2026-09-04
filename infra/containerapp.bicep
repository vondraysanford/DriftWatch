// The persistent demo endpoint: the serving container on Azure Container Apps, min replicas 0.
//
// Deployed on its own (not from main.bicep) because the image tag changes on every merge:
//   az deployment group create --resource-group DriftWatch --template-file infra/containerapp.bicep \
//     --parameters image=<acr login server>/driftwatch:<tag>
//
// Everything it depends on (environment, identity, RBAC) comes from main.bicep and is referenced
// here as existing, so this template is safe to redeploy on every push.
targetScope = 'resourceGroup'

@description('Fully qualified image reference, including the registry and tag')
param image string

@description('Region, matching the rest of the footprint')
param location string = resourceGroup().location

@description('Container Apps environment created by main.bicep')
param environmentName string = 'cae-driftwatch'

@description('User-assigned identity created by main.bicep')
param identityName string = 'id-driftwatch-app'

@description('Registry the image is pulled from')
param acrName string

@description('Storage account that receives prediction logs')
param storageAccountName string

@description('Blob container for prediction logs')
param predictionContainerName string = 'predictions'

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' existing = {
  name: environmentName
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: identityName
}

resource acr 'Microsoft.ContainerRegistry/registries@2022-12-01' existing = {
  name: acrName
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'ca-driftwatch'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
        allowInsecure: false
      }
      registries: [
        {
          server: acr.properties.loginServer
          identity: identity.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: image
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            { name: 'PREDICTION_SINK', value: 'blob' }
            { name: 'AZURE_STORAGE_ACCOUNT', value: storageAccountName }
            { name: 'PREDICTION_CONTAINER', value: predictionContainerName }
            // DefaultAzureCredential needs to be told which user-assigned identity to use.
            { name: 'AZURE_CLIENT_ID', value: identity.properties.clientId }
            { name: 'LOG_LEVEL', value: 'INFO' }
          ]
          probes: [
            {
              type: 'Readiness'
              httpGet: {
                path: '/health'
                port: 8000
              }
              initialDelaySeconds: 5
              periodSeconds: 10
              failureThreshold: 6
            }
          ]
        }
      ]
      scale: {
        // Zero idle cost is the reported number: no replica runs until a request arrives.
        minReplicas: 0
        maxReplicas: 1
        rules: [
          {
            name: 'http'
            http: {
              metadata: {
                concurrentRequests: '20'
              }
            }
          }
        ]
      }
    }
  }
}

output fqdn string = app.properties.configuration.ingress.fqdn
output url string = 'https://${app.properties.configuration.ingress.fqdn}'
