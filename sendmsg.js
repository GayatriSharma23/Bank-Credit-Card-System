const handleSendMessage = async (message) => {
  const response = await fetch('http://localhost:5000/api/chat', { // Ensure the URL is correct
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ msg: message }), // Ensure the key matches what your Flask backend expects
  });
  const data = await response.json();
  setMessages([...messages, { user: message, bot: data.response }]);
  setHistory([...history, { user: message, bot: data.response }]);
};
