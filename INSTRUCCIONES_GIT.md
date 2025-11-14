# Instrucciones para Configurar Git y Subir a GitHub

## Paso 1: Instalar Git

Si Git no está instalado, descárgalo e instálalo desde:
**https://git-scm.com/download/win**

Durante la instalación, acepta las opciones por defecto.

## Paso 2: Verificar la Instalación

Abre PowerShell o CMD y ejecuta:
```bash
git --version
```

Deberías ver algo como: `git version 2.x.x`

## Paso 3: Configurar Git (Primera vez)

Configura tu nombre y email (reemplaza con tus datos):
```bash
git config --global user.name "Tu Nombre"
git config --global user.email "tu.email@example.com"
```

## Paso 4: Conectar con el Repositorio de GitHub

Ejecuta estos comandos en PowerShell desde la carpeta del proyecto:

```bash
# Navegar a la carpeta del proyecto
cd "C:\Users\danie\Desktop\AnaliticaProyect"

# Inicializar repositorio Git
git init

# Agregar el repositorio remoto de GitHub
git remote add origin https://github.com/diazdiazdaniel59-droid/Daniel.diaz.git

# Verificar que el remoto fue agregado
git remote -v
```

## Paso 5: Agregar Archivos y Hacer Commit

```bash
# Agregar todos los archivos al staging
git add .

# Hacer el commit inicial
git commit -m "Initial commit"

# Cambiar la rama principal a 'main' (si es necesario)
git branch -M main
```

## Paso 6: Subir al Repositorio de GitHub

```bash
# Subir al repositorio remoto
git push -u origin main
```

**Nota:** Si te pide credenciales, usa tu token de acceso personal de GitHub. Puedes crear uno en:
GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)

---

## Alternativa: Usar el Script Automático

Una vez que Git esté instalado, simplemente ejecuta:

```bash
cd "C:\Users\danie\Desktop\AnaliticaProyect"
.\configurar_git.ps1
```

Luego sigue con los pasos 5 y 6 manualmente.

---

## Solución de Problemas

### Si el repositorio remoto ya existe:
```bash
git remote remove origin
git remote add origin https://github.com/diazdiazdaniel59-droid/Daniel.diaz.git
```

### Si hay un error de autenticación:
- Usa un Personal Access Token en lugar de tu contraseña
- Configura Git Credential Manager: `git config --global credential.helper manager-core`

### Para verificar el estado:
```bash
git status
```

