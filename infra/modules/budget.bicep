// Budget alert on the resource group: emails at 50/80/100% of actual monthly spend.
// Notifies only; it does not cap spending.

@description('Monthly amount in USD')
param budgetAmount int

@description('Alert recipient')
param alertEmail string

// Azure refuses to change a budget's start date once it exists ("Start date of budgets
// cannot be updated"), so this must be a stable value, not utcNow(). Pinned to the month
// the budget was first created. After a full teardown, pass the first of the current month.
@description('First day of the budget period (yyyy-MM-01). Must not change for an existing budget.')
param startDate string

resource budget 'Microsoft.Consumption/budgets@2023-11-01' = {
  name: 'budget-driftwatch'
  properties: {
    category: 'Cost'
    amount: budgetAmount
    timeGrain: 'Monthly'
    timePeriod: {
      startDate: startDate
    }
    notifications: {
      actual50Percent: {
        enabled: true
        operator: 'GreaterThanOrEqualTo'
        threshold: 50
        thresholdType: 'Actual'
        contactEmails: [
          alertEmail
        ]
      }
      actual80Percent: {
        enabled: true
        operator: 'GreaterThanOrEqualTo'
        threshold: 80
        thresholdType: 'Actual'
        contactEmails: [
          alertEmail
        ]
      }
      actual100Percent: {
        enabled: true
        operator: 'GreaterThanOrEqualTo'
        threshold: 100
        thresholdType: 'Actual'
        contactEmails: [
          alertEmail
        ]
      }
    }
  }
}
