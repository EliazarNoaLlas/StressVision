#!/bin/bash
# ============================================================================
# Nombre del archivo: backup.sh
# Propósito: Script de backup automático de la base de datos StressVision
# Autor: Equipo de Desarrollo StressVision
# Fecha de creación: 22/10/2025
# Empresa/Organización: GLORIA S.A. - StressVision Project
# ============================================================================
# Descripción:
# Este script realiza backups automáticos de la base de datos SQLite del
# sistema StressVision. Incluye funcionalidades de:
# - Backup completo de la base de datos
# - Compresión con gzip para ahorrar espacio
# - Rotación de backups antiguos
# - Verificación de integridad
# - Envío de notificaciones (opcional)
# - Logging de operaciones
#
# Uso:
#   ./backup.sh                      # Backup normal
#   ./backup.sh --compress          # Backup comprimido
#   ./backup.sh --verify            # Backup con verificación
#   ./backup.sh --encrypt           # Backup encriptado (requiere openssl)
#
# Para programar backups automáticos (crontab):
#   # Backup diario a las 2 AM
#   0 2 * * * /ruta/a/database/scripts/backup.sh --compress
#
#   # Backup cada 6 horas
#   0 */6 * * * /ruta/a/database/scripts/backup.sh
# ============================================================================

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Colores para output en consola
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # Sin color

# Obtener el directorio donde está este script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Directorio base del proyecto (dos niveles arriba de scripts/)
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Directorio de la base de datos
DATABASE_DIR="$PROJECT_DIR/database"

# Archivo de base de datos
DB_FILE="$DATABASE_DIR/gloria_stress_system.db"

# Directorio donde se guardarán los backups
BACKUP_DIR="$DATABASE_DIR/backups"

# Número de días que se conservarán los backups
RETENTION_DAYS=30

# Número máximo de backups a mantener (independiente de los días)
MAX_BACKUPS=50

# Archivo de log de backups
LOG_FILE="$DATABASE_DIR/backup.log"

# Timestamp para nombrar el backup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Nombre del archivo de backup
BACKUP_FILE="gloria_stress_backup_$TIMESTAMP.db"

# Parsear argumentos de línea de comandos
COMPRESS=false
VERIFY=false
ENCRYPT=false

# ============================================================================
# FUNCIONES
# ============================================================================

# Función para imprimir mensajes con timestamp en el log
log_message() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    
    # Escribir en el archivo de log
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    # También mostrar en consola con colores
    case $level in
        "ERROR")
            echo -e "${RED}[ERROR]${NC} $message"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} $message"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} $message"
            ;;
        "INFO")
            echo -e "${BLUE}[INFO]${NC} $message"
            ;;
        *)
            echo "$message"
            ;;
    esac
}

# Función para verificar si el archivo de base de datos existe
check_database() {
    if [ ! -f "$DB_FILE" ]; then
        log_message "ERROR" "Archivo de base de datos no encontrado: $DB_FILE"
        return 1
    fi
    
    log_message "INFO" "Base de datos encontrada: $DB_FILE"
    return 0
}

# Función para crear el directorio de backups si no existe
create_backup_dir() {
    if [ ! -d "$BACKUP_DIR" ]; then
        mkdir -p "$BACKUP_DIR"
        log_message "INFO" "Directorio de backups creado: $BACKUP_DIR"
    fi
}

# Función para realizar el backup usando SQLite
perform_backup() {
    log_message "INFO" "Iniciando backup de la base de datos..."
    
    # Ruta completa del archivo de backup
    local backup_path="$BACKUP_DIR/$BACKUP_FILE"
    
    # Usar sqlite3 para hacer un backup consistente
    # El comando .backup asegura que el backup sea consistente incluso si
    # hay transacciones en curso
    sqlite3 "$DB_FILE" ".backup '$backup_path'" 2>&1
    
    # Verificar si el comando fue exitoso
    if [ $? -eq 0 ]; then
        # Obtener el tamaño del archivo de backup
        local size=$(du -h "$backup_path" | cut -f1)
        log_message "SUCCESS" "Backup creado exitosamente: $BACKUP_FILE (Tamaño: $size)"
        echo "$backup_path"
        return 0
    else
        log_message "ERROR" "Falló la creación del backup"
        return 1
    fi
}

# Función para comprimir el backup con gzip
compress_backup() {
    local backup_path=$1
    
    log_message "INFO" "Comprimiendo backup..."
    
    # Comprimir el archivo usando gzip con máxima compresión (-9)
    gzip -9 "$backup_path" 2>&1
    
    if [ $? -eq 0 ]; then
        local compressed_file="${backup_path}.gz"
        local size=$(du -h "$compressed_file" | cut -f1)
        log_message "SUCCESS" "Backup comprimido: ${BACKUP_FILE}.gz (Tamaño: $size)"
        echo "$compressed_file"
        return 0
    else
        log_message "ERROR" "Falló la compresión del backup"
        return 1
    fi
}

# Función para encriptar el backup con openssl
encrypt_backup() {
    local backup_path=$1
    
    # Verificar que openssl está instalado
    if ! command -v openssl &> /dev/null; then
        log_message "WARNING" "openssl no está instalado. Saltando encriptación."
        return 1
    fi
    
    log_message "INFO" "Encriptando backup..."
    
    # Solicitar contraseña (en producción, esto debería venir de un archivo seguro)
    # Por ahora, usar una contraseña de ejemplo
    # TODO: Implementar gestión segura de contraseñas
    local encrypted_file="${backup_path}.enc"
    
    # Encriptar usando AES-256-CBC
    openssl enc -aes-256-cbc -salt -in "$backup_path" -out "$encrypted_file" -k "CHANGE_THIS_PASSWORD" 2>&1
    
    if [ $? -eq 0 ]; then
        # Eliminar el archivo sin encriptar por seguridad
        rm "$backup_path"
        log_message "SUCCESS" "Backup encriptado: $(basename "$encrypted_file")"
        echo "$encrypted_file"
        return 0
    else
        log_message "ERROR" "Falló la encriptación del backup"
        return 1
    fi
}

# Función para verificar la integridad del backup
verify_backup() {
    local backup_path=$1
    
    # Si el backup está comprimido, descomprimirlo temporalmente
    if [[ "$backup_path" == *.gz ]]; then
        log_message "INFO" "Descomprimiendo backup para verificación..."
        local temp_file="${backup_path%.gz}.temp"
        gunzip -c "$backup_path" > "$temp_file"
        backup_path="$temp_file"
    fi
    
    log_message "INFO" "Verificando integridad del backup..."
    
    # Ejecutar integrity_check en el backup
    local result=$(sqlite3 "$backup_path" "PRAGMA integrity_check;" 2>&1)
    
    # Limpiar archivo temporal si existe
    if [ -f "${backup_path%.gz}.temp" ]; then
        rm "${backup_path%.gz}.temp"
    fi
    
    # Verificar el resultado
    if [ "$result" = "ok" ]; then
        log_message "SUCCESS" "✓ Integridad del backup verificada correctamente"
        return 0
    else
        log_message "ERROR" "✗ El backup falló la verificación de integridad: $result"
        return 1
    fi
}

# Función para limpiar backups antiguos (rotación)
rotate_backups() {
    log_message "INFO" "Iniciando rotación de backups antiguos..."
    
    # Contar número total de backups
    local total_backups=$(find "$BACKUP_DIR" -name "gloria_stress_backup_*.db*" | wc -l)
    log_message "INFO" "Total de backups actuales: $total_backups"
    
    # Eliminar backups más antiguos que RETENTION_DAYS
    log_message "INFO" "Eliminando backups mayores a $RETENTION_DAYS días..."
    local deleted_by_age=$(find "$BACKUP_DIR" -name "gloria_stress_backup_*.db*" -type f -mtime +$RETENTION_DAYS -delete -print | wc -l)
    
    if [ $deleted_by_age -gt 0 ]; then
        log_message "INFO" "Eliminados $deleted_by_age backup(s) por antigüedad"
    fi
    
    # Si aún hay más de MAX_BACKUPS, eliminar los más antiguos
    total_backups=$(find "$BACKUP_DIR" -name "gloria_stress_backup_*.db*" | wc -l)
    
    if [ $total_backups -gt $MAX_BACKUPS ]; then
        log_message "INFO" "Límite de backups ($MAX_BACKUPS) excedido. Eliminando los más antiguos..."
        
        # Calcular cuántos backups eliminar
        local to_delete=$((total_backups - MAX_BACKUPS))
        
        # Obtener y eliminar los archivos más antiguos
        find "$BACKUP_DIR" -name "gloria_stress_backup_*.db*" -type f -printf '%T+ %p\n' | \
            sort | \
            head -n $to_delete | \
            cut -d' ' -f2- | \
            xargs -r rm -v >> "$LOG_FILE" 2>&1
        
        log_message "INFO" "Eliminados $to_delete backup(s) por exceder el límite"
    fi
    
    # Contar backups finales
    total_backups=$(find "$BACKUP_DIR" -name "gloria_stress_backup_*.db*" | wc -l)
    log_message "INFO" "Total de backups después de la rotación: $total_backups"
}

# Función para calcular estadísticas de backups
backup_statistics() {
    log_message "INFO" "Generando estadísticas de backups..."
    
    # Tamaño total de backups
    local total_size=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1)
    
    # Número de backups
    local num_backups=$(find "$BACKUP_DIR" -name "gloria_stress_backup_*.db*" | wc -l)
    
    # Backup más reciente
    local latest_backup=$(find "$BACKUP_DIR" -name "gloria_stress_backup_*.db*" -type f -printf '%T+ %p\n' | sort -r | head -n 1 | cut -d' ' -f2-)
    
    # Backup más antiguo
    local oldest_backup=$(find "$BACKUP_DIR" -name "gloria_stress_backup_*.db*" -type f -printf '%T+ %p\n' | sort | head -n 1 | cut -d' ' -f2-)
    
    echo ""
    echo "=================================="
    echo "  ESTADÍSTICAS DE BACKUPS"
    echo "=================================="
    echo "Directorio: $BACKUP_DIR"
    echo "Número de backups: $num_backups"
    echo "Tamaño total: $total_size"
    
    if [ -n "$latest_backup" ]; then
        echo "Backup más reciente: $(basename "$latest_backup")"
    fi
    
    if [ -n "$oldest_backup" ]; then
        echo "Backup más antiguo: $(basename "$oldest_backup")"
    fi
    
    echo "=================================="
    echo ""
}

# Función para enviar notificación por email (opcional)
send_notification() {
    local status=$1
    local message=$2
    
    # TODO: Implementar envío de email usando mail/sendmail
    # Por ahora, solo registrar en el log
    log_message "INFO" "Notificación: $status - $message"
    
    # Ejemplo de cómo se podría implementar:
    # echo "$message" | mail -s "StressVision Backup: $status" admin@gloria.com.pe
}

# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

# Banner de inicio
echo ""
echo "=========================================="
echo "  STRESSVISION - BACKUP DE BASE DE DATOS"
echo "=========================================="
echo ""

# Parsear argumentos
while [[ $# -gt 0 ]]; do
    case $1 in
        --compress|-c)
            COMPRESS=true
            shift
            ;;
        --verify|-v)
            VERIFY=true
            shift
            ;;
        --encrypt|-e)
            ENCRYPT=true
            shift
            ;;
        --help|-h)
            echo "Uso: $0 [opciones]"
            echo ""
            echo "Opciones:"
            echo "  --compress, -c    Comprimir el backup con gzip"
            echo "  --verify, -v      Verificar integridad del backup"
            echo "  --encrypt, -e     Encriptar el backup con AES-256"
            echo "  --help, -h        Mostrar esta ayuda"
            echo ""
            echo "Ejemplo:"
            echo "  $0 --compress --verify"
            exit 0
            ;;
        *)
            log_message "WARNING" "Opción desconocida: $1"
            shift
            ;;
    esac
done

# Iniciar el proceso de backup
log_message "INFO" "========== INICIO DEL PROCESO DE BACKUP =========="
log_message "INFO" "Timestamp: $TIMESTAMP"

# Paso 1: Verificar que la base de datos existe
if ! check_database; then
    log_message "ERROR" "Abortando backup"
    send_notification "ERROR" "Backup falló: Base de datos no encontrada"
    exit 1
fi

# Paso 2: Crear directorio de backups si no existe
create_backup_dir

# Paso 3: Realizar el backup
backup_path=$(perform_backup)
if [ $? -ne 0 ]; then
    log_message "ERROR" "Abortando backup"
    send_notification "ERROR" "Backup falló durante la creación"
    exit 1
fi

# Paso 4: Encriptar si se solicitó
if [ "$ENCRYPT" = true ]; then
    backup_path=$(encrypt_backup "$backup_path")
    if [ $? -ne 0 ]; then
        log_message "WARNING" "Continuando sin encriptación"
    fi
fi

# Paso 5: Comprimir si se solicitó
if [ "$COMPRESS" = true ]; then
    backup_path=$(compress_backup "$backup_path")
    if [ $? -ne 0 ]; then
        log_message "WARNING" "Continuando sin compresión"
    fi
fi

# Paso 6: Verificar integridad si se solicitó
if [ "$VERIFY" = true ]; then
    if ! verify_backup "$backup_path"; then
        log_message "ERROR" "El backup falló la verificación de integridad"
        send_notification "WARNING" "Backup creado pero falló verificación"
        exit 1
    fi
fi

# Paso 7: Rotar backups antiguos
rotate_backups

# Paso 8: Mostrar estadísticas
backup_statistics

# Finalización exitosa
log_message "SUCCESS" "========== BACKUP COMPLETADO EXITOSAMENTE =========="
send_notification "SUCCESS" "Backup completado: $(basename "$backup_path")"

echo ""
echo "✓ Backup completado exitosamente"
echo "  Archivo: $(basename "$backup_path")"
echo "  Ubicación: $BACKUP_DIR"
echo ""

exit 0


