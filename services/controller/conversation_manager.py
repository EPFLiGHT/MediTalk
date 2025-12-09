import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .data_models import Conversation, ConversationMessage

logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Conversation storage with JSON file persistence.
    
    Thread-safe for concurrent requests.
    """
    
    def __init__(self, storage_dir: str = "./conversations", verbose: bool = False):
        self._conversations: Dict[str, Conversation] = {}  # In-memory cache
        self._lock = threading.Lock()
        self.verbose = verbose
        self.storage_dir = Path(storage_dir)
        
        # Create storage directory
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            logger.info(f"ConversationManager initialized with storage: {self.storage_dir}")
    
    def get_conversation_path(self, conversation_id: str) -> str:
        """Get file path for a conversation JSON file"""
        return str(self.storage_dir / f"{conversation_id}.json")
    
    def _save_to_file(self, conversation: Conversation, verbose: bool = False):
        """Save conversation to JSON file"""
        file_path = self.get_conversation_path(conversation.id)
        
        with open(file_path, 'w') as f:
            json.dump(conversation.dict(), f, indent=2, default=str)
        
        if verbose:
            logger.info(f"Saved conversation {conversation.id} to {file_path}")
    
    def _load_from_file(self, conversation_id: str, verbose: bool = False) -> Optional[Conversation]:
        """Load conversation from JSON file"""
        file_path = self.get_conversation_path(conversation_id)
        
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        conversation = Conversation(**data)
        
        if verbose:
            logger.info(f"Loaded conversation {conversation_id} from {file_path}")
        
        return conversation
    
    def create_conversation(self, metadata: Optional[Dict] = None, verbose: bool = None) -> Conversation:
        """Create a new conversation and save to JSON file"""
        _verbose = verbose if verbose is not None else self.verbose
        
        conversation = Conversation(metadata=metadata)
        
        with self._lock:
            self._conversations[conversation.id] = conversation
            self._save_to_file(conversation, verbose=_verbose)
        
        if _verbose:
            logger.info(
                f"Created conversation {conversation.id} at {self.get_conversation_path(conversation.id)}"
            )
        
        return conversation
    
    def get_conversation(self, conversation_id: str, verbose: bool = None) -> Optional[Conversation]:
        """Get conversation by ID (from cache or load from file)"""
        _verbose = verbose if verbose is not None else self.verbose
        
        with self._lock:
            # Try cache first
            conversation = self._conversations.get(conversation_id)
            
            # If not in cache, try loading from file
            if not conversation:
                conversation = self._load_from_file(conversation_id, verbose=_verbose)
                if conversation:
                    self._conversations[conversation_id] = conversation
        
        if _verbose:
            if conversation:
                logger.info(f"Retrieved conversation {conversation_id} with {len(conversation.messages)} messages")
            else:
                logger.warning(f"Conversation {conversation_id} not found")
        
        return conversation
    
    def list_conversations(self, limit: int = 100) -> List[Conversation]:
        """List all conversations (most recent first)"""
        with self._lock:
            conversations = list(self._conversations.values())
        
        # Sort by updated_at descending
        conversations.sort(key=lambda c: c.updated_at, reverse=True)
        return conversations[:limit]
    
    def add_message(
        self,
        conversation_id: str,
        message: ConversationMessage,
        verbose: bool = None
    ) -> Optional[Conversation]:
        """
        Add a message to a conversation and save to file.
        
        Returns updated conversation or None if conversation doesn't exist.
        """
        _verbose = verbose if verbose is not None else self.verbose
        
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if not conversation:
                # Try loading from file
                conversation = self._load_from_file(conversation_id, verbose=_verbose)
                if conversation:
                    self._conversations[conversation_id] = conversation
            
            if not conversation:
                if _verbose:
                    logger.warning(f"Cannot add message: conversation {conversation_id} not found")
                return None
            
            conversation.messages.append(message)
            conversation.updated_at = datetime.utcnow()
            
            # Save to file
            self._save_to_file(conversation, verbose=_verbose)
            
            if _verbose:
                content_types = [c.type for c in message.content]
                logger.info(
                    f"Added {message.role} message to conversation {conversation_id}. "
                    f"Content types: {content_types}. Total messages: {len(conversation.messages)}"
                )
            
            return conversation
    
    def delete_conversation(self, conversation_id: str, verbose: bool = None) -> bool:
        """Delete a conversation (from cache and file). Returns True if deleted, False if not found."""
        _verbose = verbose if verbose is not None else self.verbose
        
        with self._lock:
            # Delete from cache
            deleted = False
            if conversation_id in self._conversations:
                del self._conversations[conversation_id]
                deleted = True
            
            # Delete file
            file_path = self.get_conversation_path(conversation_id)
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted = True
                if _verbose:
                    logger.info(f"Deleted conversation {conversation_id} (file and cache)")
            
            if not deleted and _verbose:
                logger.warning(f"Cannot delete: conversation {conversation_id} not found")
            
            return deleted
    
    def get_conversation_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[ConversationMessage]:
        """
        Get message history for a conversation.
        
        Args:
            conversation_id: Conversation ID
            limit: Optional limit on number of messages (most recent)
        
        Returns:
            List of messages (oldest first)
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
        
        messages = conversation.messages
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def clear_all(self):
        """Clear all conversations (useful for testing)"""
        with self._lock:
            self._conversations.clear()
    
    def get_stats(self) -> Dict:
        """Get statistics about stored conversations"""
        with self._lock:
            total_conversations = len(self._conversations)
            total_messages = sum(
                len(conv.messages) 
                for conv in self._conversations.values()
            )
        
        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages
        }


# Global instance
conversation_manager = ConversationManager()
