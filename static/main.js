// Enhanced message sending functionality
function sendMessage() {
    const messageInput = document.getElementById('message-input');
    const message = messageInput.value.trim();
    const errorContainer = document.getElementById('error-container');
    const loadingIndicator = document.getElementById('loading-indicator');

    // Validate input
    if (!message) {
        errorContainer.textContent = 'Please enter a message.';
        errorContainer.classList.remove('d-none');
        return;
    }

    // Clear previous errors
    errorContainer.classList.add('d-none');

    // Display user message
    displayMessage('user', message);

    // Show loading indicator
    loadingIndicator.classList.remove('d-none');

    // Get selected function
    const functionSelect = document.getElementById('function-select');
    const selectedFunction = functionSelect.value;

    // Determine API endpoint
    let url;
    switch (selectedFunction) {
        case 'search':
            url = '/search';
            break;
        case 'kbanswer':
            url = '/kbanswer';
            break;
        case 'answer':
        default:
            url = '/answer';
    }

    // Send AJAX request
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => {
        // Hide loading indicator
        loadingIndicator.classList.add('d-none');

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Display assistant's response
        displayMessage('assistant', data.message);
    })
    .catch(error => {
        // Handle errors
        console.error('Error:', error);
        errorContainer.textContent = 'Sorry, something went wrong. Please try again.';
        errorContainer.classList.remove('d-none');
        loadingIndicator.classList.add('d-none');
    });

    // Clear input field
    messageInput.value = '';
}

// Message display function
function displayMessage(sender, message) {
    const chatContainer = document.getElementById('chat-container');
    const messageDiv = document.createElement('div');

    // Add appropriate classes
    messageDiv.classList.add(
        'message', 
        `message-${sender}`, 
        'animate__animated', 
        'animate__fadeIn'
    );

    // Format message with sender and timestamp
    messageDiv.innerHTML = `
        <strong class="message-sender">${sender === 'assistant' ? 'ThinkBot' : 'You'}:</strong>
        <span class="message-content">${escapeHTML(message)}</span>
        <small class="message-timestamp">${new Date().toLocaleTimeString()}</small>
    `;

    // Append message and scroll to bottom
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Utility function to prevent XSS
function escapeHTML(str) {
    return str.replace(/[&<>'"]/g, 
        tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag] || tag));
}

// Clear chat history
function clearChatHistory() {
    const chatContainer = document.getElementById('chat-container');
    chatContainer.innerHTML = '';
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    const sendButton = document.getElementById('send-btn');
    const clearButton = document.getElementById('clear-btn');
    const messageInput = document.getElementById('message-input');

    sendButton.addEventListener('click', sendMessage);
    clearButton.addEventListener('click', clearChatHistory);

    // Enable send on Enter key
    messageInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });
});