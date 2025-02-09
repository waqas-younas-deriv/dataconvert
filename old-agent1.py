import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import os
import json
from datetime import datetime
import pickle
from pathlib import Path

class FraudDetectionAI:
    def __init__(self, historical_data_path, db_path="./new_fraud_patterns_db"):
        self.db_path = Path(db_path)
        self.stats_path = self.db_path / "pattern_stats.pkl"
        self.last_retrain_path = self.db_path / "last_retrain.txt"
        
        # Initialize pattern statistics
        self.pattern_stats = self._load_pattern_stats()
        
        # First read the CSV without type conversion
        self.data = pd.read_csv(historical_data_path)
        
        # Convert numeric columns after loading based on new data structure
        self.numeric_columns = {
            'profit_made': float,
            'number_of_trades': int,
            'volume_of_trade': float,
            'deposits': float,
            'withdrawals': float,
            'profit_per_trade': float,
            'profit_per_volume': float,
            'average_time_per_trade': float
        }
        
        # Convert each column safely
        for col, dtype in self.numeric_columns.items():
            try:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            except Exception as e:
                print(f"Warning: Error converting column {col}: {str(e)}")
        
        # Convert fraud column to boolean
        self.data['fraud'] = self.data['fraud'].map({'yes': True, 'no': False})
        
        # Remove any rows with NaN values after conversion
        self.data = self.data.dropna(subset=list(self.numeric_columns.keys()) + ['fraud'])
        
        # Initialize AI components
        self.setup_ai()
        
        # Check if retraining is needed
        if self._should_retrain():
            self.prepare_training_data()
        else:
            # Load existing vector store
            self.vector_store = Chroma(
                persist_directory=str(self.db_path),
                embedding_function=self.embeddings
            )
    
    def _load_pattern_stats(self):
        """Load pattern statistics from disk"""
        if self.stats_path.exists():
            with open(self.stats_path, 'rb') as f:
                return pickle.load(f)
        return {
            'patterns_analyzed': 0,
            'suspicious_patterns': 0,
            'common_indicators': {},
            'last_patterns': []
        }
    
    def _save_pattern_stats(self):
        """Save pattern statistics to disk"""
        self.db_path.mkdir(parents=True, exist_ok=True)
        with open(self.stats_path, 'wb') as f:
            pickle.dump(self.pattern_stats, f)
    
    def _should_retrain(self, days_threshold=7):
        """Check if the model should be retrained"""
        if not self.last_retrain_path.exists():
            return True
            
        last_retrain = datetime.fromtimestamp(self.last_retrain_path.stat().st_mtime)
        days_since_retrain = (datetime.now() - last_retrain).days
        return days_since_retrain >= days_threshold
    
    def _update_last_retrain(self):
        """Update the last retrain timestamp"""
        self.last_retrain_path.touch()
    
    def add_new_pattern(self, user_id, profit_made, num_trades, volume, deposits, withdrawals, 
                        profit_per_trade, profit_per_volume, avg_time_per_trade, is_fraudulent=True):
        """Add a new pattern to the database"""
        try:
            # Create pattern description
            pattern_description = (
                f"Trading Pattern: "
                f"User={user_id}, "
                f"Profit=${profit_made:.2f}, "
                f"Trades={num_trades}, "
                f"Volume=${volume:.2f}, "
                f"Deposits=${deposits:.2f}, "
                f"Withdrawals=${withdrawals:.2f}, "
                f"Profit/Trade=${profit_per_trade:.2f}, "
                f"Profit/Volume={profit_per_volume:.2f}, "
                f"Avg Time/Trade={avg_time_per_trade/60:.2f}min, "
                f"W/P Ratio={withdrawals/max(profit_made, 1):.2f}, "
                f"Is Fraud={is_fraudulent}"
            )
            
            # Create metadata
            metadata = {
                'user_id': str(user_id),
                'profit_made': float(profit_made),
                'number_of_trades': int(num_trades),
                'volume_of_trade': float(volume),
                'withdrawals': float(withdrawals),
                'profit_per_trade': float(profit_per_trade),
                'profit_per_volume': float(profit_per_volume),
                'average_time_per_trade': float(avg_time_per_trade),
                'withdrawal_to_profit_ratio': float(withdrawals/max(profit_made, 1)),
                'is_fraud': bool(is_fraudulent),
                'date_added': datetime.now().isoformat()
            }
            
            # Add to vector store
            self.vector_store.add_documents([
                Document(page_content=pattern_description, metadata=metadata)
            ])
            
            print(f"Successfully added new pattern to the database")
            return True
            
        except Exception as e:
            print(f"Error adding new pattern: {str(e)}")
            return False
    
    def update_pattern_stats(self, analysis_result):
        """Update pattern statistics based on analysis results"""
        self.pattern_stats['patterns_analyzed'] += 1
        
        if "SUSPICIOUS: Yes" in analysis_result.upper():
            self.pattern_stats['suspicious_patterns'] += 1
            
        # Extract reasons and update common indicators
        reasons_start = analysis_result.find("REASONS:") + 8
        reasons_end = analysis_result.find("RECOMMENDATION:")
        if reasons_start > 8 and reasons_end > 0:
            reasons = analysis_result[reasons_start:reasons_end].strip().split('\n')
            for reason in reasons:
                reason = reason.strip('- ').strip()
                self.pattern_stats['common_indicators'][reason] = \
                    self.pattern_stats['common_indicators'].get(reason, 0) + 1
        
        # Keep last 100 patterns
        self.pattern_stats['last_patterns'].append({
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis_result
        })
        self.pattern_stats['last_patterns'] = self.pattern_stats['last_patterns'][-100:]
        
        # Save updated stats
        self._save_pattern_stats()
    
    def export_fraud_patterns(self, output_file="fraud_patterns_report.json"):
        """Export fraud pattern analysis and statistics"""
        report = {
            'summary': {
                'total_patterns_analyzed': self.pattern_stats['patterns_analyzed'],
                'suspicious_patterns_detected': self.pattern_stats['suspicious_patterns'],
                'suspicious_rate': (self.pattern_stats['suspicious_patterns'] / 
                                  max(1, self.pattern_stats['patterns_analyzed'])) * 100
            },
            'common_fraud_indicators': {
                k: v for k, v in sorted(
                    self.pattern_stats['common_indicators'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]  # Top 10 indicators
            },
            'recent_suspicious_patterns': [
                pattern for pattern in self.pattern_stats['last_patterns']
                if "SUSPICIOUS: Yes" in pattern['analysis'].upper()
            ][-10:]  # Last 10 suspicious patterns
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Fraud patterns report exported to {output_file}")
        return report
    
    def prepare_training_data(self):
        """Prepare the historical data and store in vector database"""
        try:
            # Calculate additional metrics
            self.data['withdrawal_to_profit_ratio'] = self.data['withdrawals'] / self.data['profit_made'].where(self.data['profit_made'] > 0, 1)
            
            # Create documents for vector store
            documents = []
            metadatas = []
            
            for _, row in self.data.iterrows():
                try:
                    # Create a detailed description of the pattern
                    pattern_description = (
                        f"Trading Pattern: "
                        f"User={row['user_id']}, "
                        f"Profit=${row['profit_made']:.2f}, "
                        f"Trades={row['number_of_trades']}, "
                        f"Volume=${row['volume_of_trade']:.2f}, "
                        f"Withdrawals=${row['withdrawals']:.2f}, "
                        f"Profit/Trade=${row['profit_per_trade']:.2f}, "
                        f"Profit/Volume={row['profit_per_volume']:.2f}, "
                        f"Avg Time/Trade={row['average_time_per_trade']/60:.2f}min, "
                        f"W/P Ratio={row['withdrawals']/max(row['profit_made'], 1):.2f}, "
                        f"Is Fraud={row['fraud']}"
                    )
                    
                    # Store full metadata
                    metadata = {
                        'user_id': str(row['user_id']),
                        'profit_made': float(row['profit_made']),
                        'number_of_trades': int(row['number_of_trades']),
                        'volume_of_trade': float(row['volume_of_trade']),
                        'withdrawals': float(row['withdrawals']),
                        'profit_per_trade': float(row['profit_per_trade']),
                        'profit_per_volume': float(row['profit_per_volume']),
                        'average_time_per_trade': float(row['average_time_per_trade']),
                        'withdrawal_to_profit_ratio': float(row['withdrawal_to_profit_ratio']),
                        'is_fraud': bool(row['fraud'])
                    }
                    
                    documents.append(Document(
                        page_content=pattern_description,
                        metadata=metadata
                    ))
                    metadatas.append(metadata)
                    
                except (ValueError, TypeError) as e:
                    continue
            
            # Store in ChromaDB with new instance
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.db_path)
            )
            
            # Update last retrain timestamp
            self._update_last_retrain()
            
            print(f"Successfully stored {len(documents)} trading patterns in the database")
                    
        except Exception as e:
            print(f"Error in prepare_training_data: {str(e)}")
            raise
    
    def setup_ai(self):
        """Setup the AI model and embeddings"""
        self.llm = ChatOpenAI(
            model_name="o1-mini",
            api_key="sk-SMHofrEomuHPBev0pQH-UA",
            base_url="https://litellm.deriv.ai/v1",
            temperature=0.2
        )
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key="sk-SMHofrEomuHPBev0pQH-UA",
            base_url="https://litellm.deriv.ai/v1"
        )
        
        # Updated prompt template focusing on withdrawal patterns and new metrics
        self.prompt = PromptTemplate(
            input_variables=["similar_patterns", "current_activity"],
            template="""You are a fraud detection expert analyzing trading and withdrawal patterns. 

I have found these similar patterns from our database that are most relevant to the current activity:
{similar_patterns}

Please analyze the following current activity:
{current_activity}

Based on these similar patterns from our database, determine if this activity is suspicious.
Focus on:
1. Withdrawal Analysis:
   - Withdrawal amount compared to total profit (W/P Ratio)
   - Overall withdrawal pattern

2. Trading Performance:
   - Profit per trade (Is it realistic?)
   - Profit per volume (Is it consistent with market norms?)
   - Average time per trade (Are trades too quick or too slow?)

3. Volume and Activity:
   - Number of trades vs profit made
   - Trading volume vs deposits
   - Overall profitability pattern

4. Risk Indicators:
   - Compare metrics with known fraudulent and legitimate patterns
   - Look for unusual combinations of high profits with quick trades
   - Check for suspicious withdrawal timing and amounts

Provide your analysis in the following format:
SUSPICIOUS: [Yes/No]
CONFIDENCE: [0-100]%
RISK LEVEL: [High/Medium/Low]
KEY METRICS:
- W/P Ratio: [Value and Assessment]
- Profit/Trade: [Value and Assessment]
- Time/Trade: [Value and Assessment]
- Volume Pattern: [Assessment]
REASONS: [List your main reasons]
RECOMMENDATION: [Clear recommendation about whether to allow or investigate the withdrawal request]
"""
        )
    
    def get_similar_patterns(self, current_activity, num_patterns=5):
        """Find similar patterns from the vector store"""
        results = self.vector_store.similarity_search(
            current_activity,
            k=num_patterns
        )
        return "\n".join([f"Pattern {i+1}: {doc.page_content}" 
                         for i, doc in enumerate(results)])
    
    def analyze_activity(self, user_id, profit_made, num_trades, volume, withdrawals, 
                        profit_per_trade, profit_per_volume, avg_time_per_trade):
        """Use AI to analyze if the current activity is suspicious"""
        # Format the current activity with new metrics
        current_activity = (
            f"Trading Pattern: "
            f"User={user_id}, "
            f"Profit=${profit_made:.2f}, "
            f"Trades={num_trades}, "
            f"Volume=${volume:.2f}, "
            f"Withdrawals=${withdrawals:.2f}, "
            f"Profit/Trade=${profit_per_trade:.2f}, "
            f"Profit/Volume={profit_per_volume:.2f}, "
            f"Avg Time/Trade={avg_time_per_trade/60:.2f}min, "
            f"W/P Ratio={withdrawals/max(profit_made, 1):.2f}"
        )
        
        # Get similar patterns from vector store
        similar_patterns = self.get_similar_patterns(current_activity)
        
        # Create the chain and get AI analysis
        chain = self.prompt | self.llm
        response = chain.invoke({
            "similar_patterns": similar_patterns,
            "current_activity": current_activity
        })
        
        return response.content

def evaluate_withdrawal_request(agent, user_id, profit_made, num_trades, volume, withdrawals, 
                                profit_per_trade, profit_per_volume, avg_time_per_trade):
    """Get AI evaluation for a withdrawal request"""
    print("\nAnalyzing trading activity...")
    print("-" * 50)
    
    analysis = agent.analyze_activity(
        user_id, profit_made, num_trades, volume, withdrawals,
        profit_per_trade, profit_per_volume, avg_time_per_trade
    )
    
    print("\nAI Analysis:")
    print("-" * 50)
    print(analysis)
    print("-" * 50)
    
    # Extract confidence level
    try:
        confidence_line = [line for line in analysis.split('\n') if 'CONFIDENCE:' in line][0]
        confidence = int(confidence_line.split('%')[0].split(':')[1].strip())
        
        print("\nConfidence Level Interpretation:")
        if confidence >= 80:
            print("HIGH CONFIDENCE: Strong evidence of suspicious activity")
        elif confidence >= 50:
            print("MEDIUM CONFIDENCE: Some suspicious patterns detected, requires attention")
        else:
            print("LOW CONFIDENCE: Limited evidence of suspicious activity, but some unusual patterns")
    except:
        print("Could not parse confidence level")
    
    # The decision is embedded in the AI's analysis
    return "SUSPICIOUS: No" in analysis.upper()

if __name__ == "__main__":
    # Initialize the AI agent with the new historical fraud data
    agent = FraudDetectionAI('/Users/waqasyounas/Downloads/new_data.csv')
    
    print("Enhanced Fraud Detection System Initialized")
    print("\nOptions:")
    print("1. Analyze withdrawal request")
    print("2. Add new pattern to database")
    print("3. Export fraud patterns report")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ")
            
            if choice == "1":
                user_id = input("Enter user ID: ")
                profit_made = float(input("Enter total profit made: "))
                num_trades = int(input("Enter number of trades: "))
                volume = float(input("Enter trading volume: "))
                withdrawals = float(input("Enter withdrawal amount: "))
                avg_time_per_trade = float(input("Enter average time per trade (seconds): "))
                
                # Calculate profit per trade and profit per volume
                profit_per_trade = profit_made / num_trades if num_trades > 0 else 0
                profit_per_volume = profit_made / volume if volume > 0 else 0
                
                analysis = agent.analyze_activity(
                    user_id, profit_made, num_trades, volume,
                    withdrawals, profit_per_trade, profit_per_volume, avg_time_per_trade
                )
                agent.update_pattern_stats(analysis)
                
                print("\nDetailed Analysis:")
                print("-" * 50)
                print(analysis)
                print("-" * 50)
                
            elif choice == "2":
                # Add new pattern with the new metrics
                user_id = input("Enter user ID: ")
                profit_made = float(input("Enter total profit made: "))
                num_trades = int(input("Enter number of trades: "))
                volume = float(input("Enter trading volume: "))
                withdrawals = float(input("Enter withdrawal amount: "))
                avg_time_per_trade = float(input("Enter average time per trade (seconds): "))
                is_fraudulent = input("Is this a fraudulent pattern? (y/n): ").lower() == 'y'
                
                # Calculate profit per trade and profit per volume
                profit_per_trade = profit_made / num_trades if num_trades > 0 else 0
                profit_per_volume = profit_made / volume if volume > 0 else 0
                
                agent.add_new_pattern(
                    user_id, profit_made, num_trades, volume,
                    withdrawals, profit_per_trade, profit_per_volume,
                    avg_time_per_trade, is_fraudulent
                )
                
            elif choice == "3":
                report = agent.export_fraud_patterns()
                print("\nFraud Patterns Summary:")
                print(f"Total Patterns Analyzed: {report['summary']['total_patterns_analyzed']}")
                print(f"Suspicious Patterns: {report['summary']['suspicious_patterns_detected']}")
                print(f"Suspicious Rate: {report['summary']['suspicious_rate']:.2f}%")
                
            elif choice == "4":
                break
                
        except ValueError:
            print("Please enter valid numbers")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    print("\nThank you for using the Enhanced Fraud Detection System")
