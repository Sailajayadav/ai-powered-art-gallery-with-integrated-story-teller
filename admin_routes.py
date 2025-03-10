from flask import Blueprint, request, jsonify, session, render_template, abort, current_app
from functools import wraps
from datetime import datetime
import json
import os
# Create blueprint for admin routes
admin_blueprint = Blueprint('admin', __name__, url_prefix='/admin')

# In-memory storage for gallery requests (in production, use a database)
# Format: publication_requests.json
# [
#   {
#     "id": "request_id",
#     "username": "username",
#     "gallery_name": "Gallery Name",
#     "submission_date": "YYYY-MM-DD HH:MM:SS",
#     "status": "pending|approved|rejected",
#     "admin_notes": "notes from admin",
#     "admin_decision_date": "YYYY-MM-DD HH:MM:SS",
#     "admin_username": "admin who made decision"
#   }
# ]

def load_requests():
    """Load publication requests from JSON file"""
    try:
        if os.path.exists('publication_requests.json'):
            with open('publication_requests.json', 'r') as f:
                return json.load(f)
        else:
            return []
    except Exception as e:
        current_app.logger.error(f"Error loading requests: {str(e)}")
        return []

def save_requests(requests):
    """Save publication requests to JSON file"""
    try:
        with open('publication_requests.json', 'w') as f:
            json.dump(requests, f, indent=2)
    except Exception as e:
        current_app.logger.error(f"Error saving requests: {str(e)}")

# Admin authentication middleware
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user') or not session.get('is_admin', False):
            if request.content_type == 'application/json':
                return jsonify({'error': 'Unauthorized access'}), 401
            return abort(401)
        return f(*args, **kwargs)
    return decorated_function

# Admin dashboard route for viewing publication requests
@admin_blueprint.route('/dashboard')
@admin_required
def admin_dashboard():
    """Render the admin dashboard"""
    return render_template('admin-dashboard.html')

# API endpoint to get all publication requests
@admin_blueprint.route('/api/publication-requests')
@admin_required
def get_publication_requests():
    """Get all publication requests"""
    requests = load_requests()
    
    # Filter by status if specified
    status = request.args.get('status')
    if status:
        requests = [req for req in requests if req['status'] == status]
    
    # Sort by submission date (newest first)
    requests.sort(key=lambda x: x['submission_date'], reverse=True)
    
    return jsonify(requests)

# API endpoint to get a specific publication request
@admin_blueprint.route('/api/publication-requests/<request_id>')
@admin_required
def get_publication_request(request_id):
    """Get a specific publication request"""
    requests = load_requests()
    request_data = next((req for req in requests if req['id'] == request_id), None)
    
    if not request_data:
        return jsonify({'error': 'Request not found'}), 404
    
    return jsonify(request_data)

# API endpoint to approve a publication request
@admin_blueprint.route('/api/publication-requests/<request_id>/approve', methods=['POST'])
@admin_required
def approve_request(request_id):
    """Approve a publication request"""
    admin_notes = request.json.get('admin_notes', '')
    
    requests = load_requests()
    request_index = next((i for i, req in enumerate(requests) if req['id'] == request_id), None)
    
    if request_index is None:
        return jsonify({'error': 'Request not found'}), 404
    
    # Update request status
    requests[request_index]['status'] = 'approved'
    requests[request_index]['admin_notes'] = admin_notes
    requests[request_index]['admin_decision_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    requests[request_index]['admin_username'] = session.get('user')
    
    save_requests(requests)
    
    # In a real application, you would also:
    # 1. Update the gallery's publication status in your database
    # 2. Notify the user about the approval
    # 3. Make the gallery publicly accessible
    
    return jsonify({
        'success': True,
        'message': f"Gallery '{requests[request_index]['gallery_name']}' has been approved and published"
    })

# API endpoint to reject a publication request
@admin_blueprint.route('/api/publication-requests/<request_id>/reject', methods=['POST'])
@admin_required
def reject_request(request_id):
    """Reject a publication request"""
    admin_notes = request.json.get('admin_notes', '')
    
    if not admin_notes:
        return jsonify({'error': 'Admin notes are required when rejecting a publication request'}), 400
    
    requests = load_requests()
    request_index = next((i for i, req in enumerate(requests) if req['id'] == request_id), None)
    
    if request_index is None:
        return jsonify({'error': 'Request not found'}), 404
    
    # Update request status
    requests[request_index]['status'] = 'rejected'
    requests[request_index]['admin_notes'] = admin_notes
    requests[request_index]['admin_decision_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    requests[request_index]['admin_username'] = session.get('user')
    
    save_requests(requests)
    
    # In a real application, you would also:
    # 1. Update the gallery's publication status in your database
    # 2. Notify the user about the rejection and provide feedback
    
    return jsonify({
        'success': True,
        'message': f"Gallery '{requests[request_index]['gallery_name']}' has been rejected"
    })

# API endpoint to get publication request statistics
@admin_blueprint.route('/api/statistics')
@admin_required
def get_statistics():
    """Get statistics about publication requests"""
    requests = load_requests()
    
    pending_count = sum(1 for req in requests if req['status'] == 'pending')
    approved_count = sum(1 for req in requests if req['status'] == 'approved')
    rejected_count = sum(1 for req in requests if req['status'] == 'rejected')
    total_count = len(requests)
    
    return jsonify({
        'pending': pending_count,
        'approved': approved_count,
        'rejected': rejected_count,
        'total': total_count
    })

# Route for creating a test publication request (for development purposes)
@admin_blueprint.route('/api/create-test-request', methods=['POST'])
def create_test_request():
    """Create a test publication request (development only)"""
    if current_app.env != 'development':
        return jsonify({'error': 'This endpoint is only available in development mode'}), 403
    
    requests = load_requests()
    
    new_request = {
        'id': f"request_{len(requests) + 1}",
        'username': request.json.get('username', 'test_user'),
        'gallery_name': request.json.get('gallery_name', 'Test Gallery'),
        'submission_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'pending',
        'admin_notes': '',
        'admin_decision_date': None,
        'admin_username': None
    }
    
    requests.append(new_request)
    save_requests(requests)
    
    return jsonify({
        'success': True,
        'message': 'Test request created',
        'request': new_request
    })