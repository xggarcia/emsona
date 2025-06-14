from flask import Flask, render_template, request, redirect, url_for, jsonify
import re
import csv
import os
import subprocess
import json
import sys
import threading
import uuid
import time
import sqlite3
from datetime import datetime

# Fix path issues for production
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from recommendator.app.app import recomendator_with_faiss

app = Flask(__name__)

# Configuration for production
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')

# Diccionario global para almacenar el estado de los procesos
processing_status = {}

# Database initialization
def init_db():
    """Initialize the SQLite database for storing likes"""
    # Use absolute path for database
    db_path = os.path.join(os.path.dirname(__file__), 'likes.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create likes table if it doesn't exist - NO UNIQUE constraint
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS likes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            titulo TEXT,
            artista TEXT,
            link TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def add_like(video_id, titulo=None, artista=None, link=None):
    """Add a like to the database - allows multiple likes"""
    db_path = os.path.join(os.path.dirname(__file__), 'likes.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Insert a new like record every time (no checking for duplicates)
        cursor.execute('''
            INSERT INTO likes (video_id, titulo, artista, link)
            VALUES (?, ?, ?, ?)
        ''', (video_id, titulo, artista, link))
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error adding like: {e}")
        return False
    finally:
        conn.close()

def get_liked_songs():
    """Get all liked songs from the database with proper like counts"""
    # Fix: Use absolute path like in other functions
    db_path = os.path.join(os.path.dirname(__file__), 'likes.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get the count of likes per video_id and take the most recent record for each video
        cursor.execute('''
            SELECT 
                video_id, 
                titulo, 
                artista, 
                link, 
                COUNT(*) as likes,
                MAX(timestamp) as last_liked
            FROM likes
            GROUP BY video_id
            ORDER BY likes DESC, last_liked DESC
        ''')
        
        results = cursor.fetchall()
        
        canciones_gustadas = []
        for row in results:
            video_id, titulo, artista, link, likes, last_liked = row
            canciones_gustadas.append({
                "titulo": titulo or "Título desconocido",
                "artista": artista or "Artista desconocido", 
                "likes": likes,
                "link": link or f"https://www.youtube.com/watch?v={video_id}"
            })
        
        return canciones_gustadas
    except Exception as e:
        print(f"Error getting liked songs: {e}")
        return []
    finally:
        conn.close()

def get_song_info_from_csv(video_id):
    """Get song information from CSV metadata file"""
    try:
        csv_path = os.path.join(app.static_folder, 'csv', 'catalan_music_metadata.csv')
        with open(csv_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                if row.get('video_id') == video_id:
                    return {
                        'titulo': row.get('song_title', 'Título desconocido'),
                        'artista': row.get('channel_name', 'Artista desconocido')
                    }
    except Exception as e:
        print(f"Error reading CSV for video {video_id}: {e}")
    
    return {'titulo': 'Título desconocido', 'artista': 'Artista desconocido'}

# Initialize database when the app starts
init_db()

# Función para extraer el ID de video de YouTube de una URL
def obtener_video_id(url):
    if not url:
        return None
    
    # Asegurar que la URL tenga el protocolo para el procesamiento correcto
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        # Método 1: Extraer usando expresiones regulares para varios formatos de URL
        patron = r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
        match = re.search(patron, url)
        if match:
            return match.group(1)
        
        # Método 2: Extraer de la URL de watch usando parámetros
        if 'youtube.com/watch' in url:
            from urllib.parse import urlparse, parse_qs
            parsed_url = urlparse(url)
            video_id = parse_qs(parsed_url.query).get('v')
            if video_id and len(video_id[0]) == 11:
                return video_id[0]
        
        # Método 3: Extraer de URL corta youtu.be
        if 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[1].split('?')[0].split('&')[0]
            if len(video_id) == 11:
                return video_id
        
        return None
    except Exception as e:
        print(f"Error al extraer ID de video: {e}")
        return None

def is_single_video(url):
    """
    Check if the URL is for a single video or a playlist.
    
    Args:
        url (str): YouTube URL to check
        
    Returns:
        bool: True if it's a single video, False if it's a playlist or invalid URL
    """
    if not url or not isinstance(url, str):
        return False
    
    # Ensure URL has protocol
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
        
    try:
        # Check for playlist indicators (más exhaustivo)
        playlist_indicators = [
            'playlist?',
            '&list=',
            '?list=',
            '/playlist/',
            'list=PL',  # Playlists típicamente empiezan con PL
            'list=UU',  # Channel uploads playlist
            'list=FL',  # Favorites playlist
            'list=LL',  # Liked videos playlist
            'list=WL',  # Watch later playlist
        ]
        
        # If any playlist indicator is found, return False
        if any(indicator in url for indicator in playlist_indicators):
            return False
            
        # Check if it's a valid video URL using the existing obtener_video_id function
        video_id = obtener_video_id(url)
        if video_id is None:
            return False
            
        # Additional check: ensure it's not a channel URL
        channel_indicators = [
            '/channel/',
            '/user/',
            '/c/',
            '@'  # New YouTube handle format
        ]
        
        if any(indicator in url for indicator in channel_indicators):
            return False
            
        return True
        
    except Exception as e:
        print(f"Error checking URL type: {e}")
        return False

# Función para procesar la canción en segundo plano
def process_song_background(process_id, youtube_url):
    """Procesa la canción en segundo plano y actualiza el estado"""
    try:
        # Definir las rutas necesarias
        song_path = "./recommendator/app/store_song"
        embedding_path = "./recommendator/app/store_embedding"
        model_path = "./recommendator/app/MODEL/model.pb"
        metadata_path = "./recommendator/app/METADATA/metadata.csv"
        store_metadata_path = "./recommendator/app/store_metadata"
        song_embeddings = "./recommendator/app/EMBEDDINGS/embeddings.pkl"
        store_svg = "./recommendator/app/SVG/plot.svg"
        k = 1
        n = 6

        # ✅ Llamar a la función recomendator_with_faiss (ahora devuelve dict)
        result = recomendator_with_faiss(
            song_path, youtube_url, embedding_path, model_path, metadata_path, store_metadata_path, song_embeddings, store_svg, k, n
        )

        # ✅ Extraer recomendaciones y coordenadas del resultado
        recommendations = result['recommendations']
        coordinates = result['coordinates']

        # Construir la lista de canciones similares
        canciones_similares = []
        for index, rec in recommendations.iterrows():
            # Skip the first recommendation (index 0)
            if index == 0:
                continue
                
            # Convert distance to percentage similarity
            distance = float(rec.get('Distance', 0))
            if distance < 5:
                similarity = 100
            elif distance > 30:
                similarity = 0
            else:
                # Linear interpolation between 100% at distance=5 and 0% at distance=30
                similarity = 100 - ((distance - 5) * (100 / 25))
                
            cancion = {
                "titulo": rec.get("Song", "Título desconocido"),
                "artista": rec.get("Artist", "Desconocido"),
                "similitud": f"{similarity:.0f}%",  # Format as percentage without decimals
                "link": rec.get("YT Link", ""),
                "id": rec.get("video_id", "Z5LVw2abUlw")
            }
            canciones_similares.append(cancion)

        # ✅ Actualizar el estado como completado con coordenadas
        processing_status[process_id] = {
            'status': 'completed',
            'result': canciones_similares,
            'coordinates': coordinates,
            'video_id': obtener_video_id(youtube_url)
        }
        
    except Exception as e:
        print(f"Error al procesar la canción: {e}")
        processing_status[process_id] = {
            'status': 'error',
            'error': str(e),
            'video_id': obtener_video_id(youtube_url)
        }

# Función para leer el archivo CSV de canciones
def leer_canciones_csv():
    canciones = []
    try:
        csv_path = os.path.join(app.static_folder, 'csv', 'catalan_music_metadata.csv')
        with open(csv_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                canciones.append(row)
        return canciones
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return []

def get_likes_count_for_video(video_id):
    """Get the number of likes for a specific video"""
    conn = sqlite3.connect('likes.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT COUNT(*) as likes
            FROM likes
            WHERE video_id = ?
        ''', (video_id,))
        
        result = cursor.fetchone()
        return result[0] if result else 0
    except Exception as e:
        print(f"Error getting likes count for video {video_id}: {e}")
        return 0
    finally:
        conn.close()

@app.route('/')
def index():
    error_type = request.args.get('error')
    error_message = None
    
    if error_type == 'playlist':
        error_message = "Si us plau, envia l'enllaç d'un vídeo únic, no d'una playlist de YouTube."
    elif error_type == 'invalid':
        error_message = "L'enllaç de YouTube no és vàlid. Si us plau, verifica que sigui correcte."
    elif error_type == 'empty':
        error_message = "Si us plau, envia un enllaç de YouTube vàlid."
    
    return render_template('index.html', pagina_activa='inicio', error_message=error_message)


@app.route('/loading', methods=['POST'])
def loading():
    youtube_url = request.form.get('youtube_url')
    
    # ✅ Validaciones de URL
    if not youtube_url or not youtube_url.strip():
        return redirect(url_for('index', error='empty'))
    
    # Verificar que contenga youtube.com o youtu.be
    if 'youtube.com' not in youtube_url and 'youtu.be' not in youtube_url:
        return redirect(url_for('index', error='invalid'))
    
    # Validar que sea un video único, no una playlist
    if not is_single_video(youtube_url):
        return redirect(url_for('index', error='playlist'))
    
    video_id = obtener_video_id(youtube_url)
    
    # Verificar que se pudo extraer el video ID
    if not video_id:
        return redirect(url_for('index', error='invalid'))
    
    # Generar un ID único para este proceso
    process_id = str(uuid.uuid4())
    
    # Inicializar el estado del proceso
    processing_status[process_id] = {
        'status': 'processing',
        'video_id': video_id
    }
    
    # Iniciar el procesamiento en segundo plano
    thread = threading.Thread(target=process_song_background, args=(process_id, youtube_url))
    thread.daemon = True
    thread.start()
    
    print(f"URL recibida: {youtube_url}")
    print(f"ID extraído: {video_id}")
    print(f"Process ID: {process_id}")
    
    # Renderizar la página de loading con el process_id
    return render_template('loading.html', video_id=video_id, process_id=process_id, pagina_activa='inicio')

@app.route('/api/check_status/<process_id>')
def check_status(process_id):
    """API endpoint para verificar el estado del procesamiento"""
    if process_id not in processing_status:
        return jsonify({'status': 'not_found'})
    
    return jsonify(processing_status[process_id])

@app.route('/api/valoraciones', methods=['POST'])
def api_valoraciones():
    """API endpoint to handle rating submissions and likes"""
    try:
        data = request.get_json()
        
        video_to_like = data.get('videoUsuario')  # This is now the video to be liked
        video_recomendacion = data.get('videoRecomendacion') 
        valor_cancion = int(data.get('valorCancion', 0))
        valor_recomendacion = int(data.get('valorRecomendacion', 0))
        
        print(f"=== DEBUG RATING ===")
        print(f"Video to like: {video_to_like}")
        print(f"Recommended video: {video_recomendacion}")
        print(f"Song rating: {valor_cancion}, Recommendation rating: {valor_recomendacion}")
        print(f"Raw data received: {data}")
        print(f"===================")
        
        # If user liked the song (valorCancion = 1), add it to likes
        if valor_cancion == 1 and video_to_like:
            song_info = get_song_info_from_csv(video_to_like)
            print(f"Song info found: {song_info}")
            link = f"https://www.youtube.com/watch?v={video_to_like}"
            
            success = add_like(
                video_id=video_to_like,
                titulo=song_info['titulo'],
                artista=song_info['artista'],
                link=link
            )
            
            if success:
                print(f"✅ Added like for video: {video_to_like} - {song_info['titulo']}")
            else:
                print(f"❌ Failed to add like for video: {video_to_like}")
        elif valor_cancion == 0:
            print(f"User did not like the song: {video_to_like} - no like added")
        
        return jsonify({
            'status': 'success',
            'message': 'Valoración registrada correctamente'
        })
        
    except Exception as e:
        print(f"Error processing rating: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/results')
def results():
    # Obtener el process_id y video_id desde la URL
    process_id = request.args.get('process_id')
    video_id = request.args.get('video_id', 'Z5LVw2abUlw')
    
    print(f"Process ID recibido en resultados: {process_id}")
    print(f"Video ID recibido en resultados: {video_id}")
    
    canciones_similares = []
    coordinates = []
    
    if process_id and process_id in processing_status:
        status_data = processing_status[process_id]
        if status_data['status'] == 'completed':
            canciones_similares = status_data.get('result', [])
            coordinates = status_data.get('coordinates', [])
            # Use the video_id from processing_status, not from URL
            video_id = status_data.get('video_id', video_id)
            
            # Add likes count to each recommendation and sort by likes
            for cancion in canciones_similares:
                cancion['likes'] = get_likes_count_for_video(cancion['id'])
                print(f"Song: {cancion['titulo']} - Likes: {cancion['likes']}")
            
            # Sort recommendations by likes count (descending order)
            canciones_similares.sort(key=lambda x: x['likes'], reverse=True)
            print("Recommendations sorted by likes count")
    
    return render_template(
        'results.html',
        canciones=canciones_similares,
        coordinates=coordinates,
        video_id_usuario=video_id,
        pagina_activa='inicio'
    )

@app.route('/library')
def library():
    # Obtener parámetros de búsqueda y paginación
    query = request.args.get('query', '').lower()
    page = int(request.args.get('page', 1))
    per_page = 10
    
    # Leer todas las canciones del CSV
    todas_canciones = leer_canciones_csv()
    
    # Filtrar por búsqueda si hay query
    if query:
        canciones_filtradas = [
            cancion for cancion in todas_canciones 
            if query in cancion['song_title'].lower() or 
               query in cancion['channel_name'].lower() or 
               query in cancion['original_title'].lower()
        ]
    else:
        canciones_filtradas = todas_canciones
    
    # Calcular total de páginas
    total_canciones = len(canciones_filtradas)
    total_paginas = (total_canciones + per_page - 1) // per_page
    
    # Obtener canciones para la página actual
    inicio = (page - 1) * per_page
    fin = min(inicio + per_page, total_canciones)
    canciones_pagina = canciones_filtradas[inicio:fin]
    
    return render_template('library.html', 
                          canciones=canciones_pagina,
                          query=query,
                          page=page,
                          total_paginas=total_paginas,
                          total_canciones=total_canciones,
                          pagina_activa='biblioteca')

@app.route('/about')
def about():
    return render_template('about.html', pagina_activa='nosotros')

@app.route('/liked')
def liked():
    # Get liked songs from database instead of hardcoded data
    canciones_gustadas = get_liked_songs()
    
    # If no songs in database, show default examples
    if not canciones_gustadas:
        canciones_gustadas = [ ]
    
    return render_template('liked.html', canciones=canciones_gustadas, pagina_activa='liked')

@app.route('/forms', methods=['GET', 'POST'])
def suggest_song():
    error = None

    if request.method == 'POST':
        title  = request.form.get('title', '').strip()
        artist = request.form.get('artist', '').strip()
        genre  = request.form.get('genre', '').strip()

        # simple validation
        if not title or not artist or not genre:
            error = "Por favor completa todos los campos obligatorios."
            return render_template('forms.html', error=error)

        # aquí guardarías la sugerencia en tu base de datos...
        # por ejemplo:
        # db.save({ 'title': title, 'artist': artist, 'album': request.form.get('album'), ... })

        return redirect(url_for('index'))

    # GET → muestro el formulario vacío
    return render_template('forms.html', error=error)

if __name__ == '__main__':
    # For production deployment
    port = int(os.environ.get('PORT', 8080))  # DigitalOcean uses 8080
    app.run(host='0.0.0.0', port=port, debug=False)