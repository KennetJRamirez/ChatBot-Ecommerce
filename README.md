# ChatBot-Ecommerce

## ¿Qué hace este proyecto?

Este chatbot permite que el usuario ingrese lo que desea buscar en ropa, incluyendo atributos como tipo, color, genero. 
El sistema procesa la consulta y devuelve las **top 3 mejores sugerencias** de productos que se ajustan a lo que el usuario quiere comprar.
Dando como resultado salidas que incluyen tanto el nombre de la prenda, su descripcion, precio y nivel de % de acierto de la sugerencia.
---

##  Pasos para instalar y ejecutar

```bash
git clone https://github.com/KennetJRamirez/ChatBot-Ecommerce.git
cd ChatBot-Ecommerce

# Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutando con uvicorn 
uvicorn app:app --reload
