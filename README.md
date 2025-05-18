# Stamps 3 Application

## Starting the Application

```bash
# Build with no cache to ensure fresh dependencies
docker-compose build --no-cache

# Start all services in detached mode
docker-compose up -d
```

## Stopping the Application

```bash
# Stop and remove containers, networks, and volumes
docker-compose down -v
```

## Monitoring

```bash
# Check running containers
docker ps -a

# Check service status
docker-compose ps

# View database logs
docker logs mysql_container

# View backend logs
docker logs fastapi_backend
```

## Access

- Backend API: http://localhost:8000/docs
- Frontend: http://localhost:5173

## Testing db

```bash
docker-compose exec backend python /app/run_tests_docker.py
```

## Troubleshooting Database Issues

If you encounter database errors during initialization:

### Complete Reset

```bash
# Stop and completely remove everything including volumes
docker-compose down -v

# Remove any residual Docker volumes
docker volume rm stamps_3_db_data

# Rebuild and restart
docker-compose build --no-cache backend
docker-compose up -d
```

### Common Errors

1. **Unknown column errors**: Caused by mismatches between the model and database schema
   - Solution: Complete reset as described above

2. **Duplicate entry errors**: Duplicate values in unique columns
   - Solution: Modify docker-compose.yml to set `RESET_DB=true` if not already set

3. **Type conversion errors**: Data type mismatches 
   - Solution: Check data formats in CSV files or fix models

### Manually Checking Database

```bash
# Connect to MySQL container
docker exec -it mysql_container mysql -u rbonkass -p

# Enter password when prompted (from .env file)

# Inside MySQL
USE stamps_db;
SHOW TABLES;
DESCRIBE sets;  # View table structure
SELECT * FROM sets LIMIT 5;  # Check data
```