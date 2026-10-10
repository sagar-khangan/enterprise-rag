# Database Connection Pool Incident

## Summary
The application experienced elevated API latency because
database connections were exhausted during peak traffic.

## Root Cause
A sudden increase in concurrent requests exhausted the
database connection pool.

The maximum pool size was configured to 20 connections
while the application had more than 100 concurrent requests.

## Resolution
The engineering team introduced connection pooling
improvements and configured request concurrency limits.

## Prevention
1. Monitor active database connections.
2. Configure alerts for pool utilization above 80%.
3. Add connection acquisition timeouts.
4. Review database capacity during load testing.

## Owner
Platform Engineering Team.