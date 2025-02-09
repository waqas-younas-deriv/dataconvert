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
    def __init__(self, historical_data_path, db_path="./fraud_patterns_db"):
        self.db_path = Path(db_path)
        self.stats_path = self.db_path / "pattern_stats.pkl"
        self.last_retrain_path = self.db_path / "last_retrain.txt"
        
        # Initialize pattern statistics
        self.pattern_stats = self._load_pattern_stats()
        
        # First read the CSV without type conversion
        self.data = pd.read_csv(historical_data_path)
        
        # Convert numeric columns after loading
        self.numeric_columns = {
            'Client P&L': float,
            'Trades Number': int,
            'Trading Volume': float,
            'Deposits': float,
            'Withdrawals': float
        }
        
        # Convert each column safely
        for col, dtype in self.numeric_columns.items():
            try:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            except Exception as e:
                print(f"Warning: Error converting column {col}: {str(e)}")
        
        # Remove any rows with NaN values after conversion
        self.data = self.data.dropna(subset=self.numeric_columns.keys())
        
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
    
    def add_new_pattern(self, deposit, trading_volume, num_trades, pnl, is_fraudulent=True):
        """Add a new pattern to the database"""
        try:
            # Calculate metrics
            pnl_percentage = (pnl / trading_volume) * 100
            volume_per_trade = trading_volume / num_trades
            
            # Create pattern description
            pattern_description = (
                f"Trading Pattern: "
                f"Deposit=${deposit}, "
                f"Volume=${trading_volume}, "
                f"Trades={num_trades}, "
                f"PNL=${pnl}, "
                f"PNL%={pnl_percentage:.2f}%"
            )
            
            # Create metadata
            metadata = {
                'deposit': float(deposit),
                'trading_volume': float(trading_volume),
                'num_trades': int(num_trades),
                'pnl': float(pnl),
                'pnl_percentage': float(pnl_percentage),
                'volume_per_trade': float(volume_per_trade),
                'is_fraudulent': is_fraudulent,
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
            # Calculate metrics safely
            self.data['PNL_percentage'] = (self.data['Client P&L'] / self.data['Trading Volume']) * 100
            self.data['Volume_per_trade'] = self.data['Trading Volume'] / self.data['Trades Number']
            
            # Create documents for vector store
            documents = []
            metadatas = []
            
            for _, row in self.data.iterrows():
                try:
                    # Create a detailed description of the pattern
                    pattern_description = (
                        f"Trading Pattern: Deposit=${row['Deposits']}, "
                        f"Volume=${row['Trading Volume']}, "
                        f"Trades={row['Trades Number']}, "
                        f"PNL=${row['Client P&L']}, "
                        f"PNL%={(row['Client P&L']/row['Trading Volume'])*100:.2f}%"
                    )
                    
                    # Store full metadata
                    metadata = {
                        'deposit': float(row['Deposits']),
                        'trading_volume': float(row['Trading Volume']),
                        'num_trades': int(row['Trades Number']),
                        'pnl': float(row['Client P&L']),
                        'pnl_percentage': float(row['PNL_percentage']),
                        'volume_per_trade': float(row['Volume_per_trade'])
                    }
                    
                    documents.append(Document(
                        page_content=pattern_description,
                        metadata=metadata
                    ))
                    metadatas.append(metadata)
                    
                except (ValueError, TypeError) as e:
                    continue
            
            # Store in ChromaDB
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory="./fraud_patterns_db"
            )
            
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
        
        # Create a prompt template that includes historical patterns
        self.prompt = PromptTemplate(
            input_variables=["similar_patterns", "current_activity"],
            template="""You are a fraud detection expert analyzing trading patterns. 

I have found these similar patterns from our fraud database that are most relevant to the current activity:
{similar_patterns}

Please analyze the following current trading activity:
{current_activity}

Based on these similar patterns from our database, determine if this activity is suspicious.
Focus on:
1. If the PNL percentage is suspiciously low (less than 5%) compared to trading volume
2. If the number of trades and trading volume pattern matches suspicious historical patterns
3. If there are any other red flags based on the similar patterns shown above

Provide your analysis in the following format:
SUSPICIOUS: [Yes/No]
CONFIDENCE: [0-100]%
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
    
    def analyze_activity(self, deposit, trading_volume, num_trades, current_pnl):
        """Use AI to analyze if the current activity is suspicious"""
        # Format the current activity
        current_activity = (
            f"Trading Pattern: "
            f"Deposit=${deposit}, "
            f"Volume=${trading_volume}, "
            f"Trades={num_trades}, "
            f"PNL=${current_pnl}, "
            f"PNL%={(current_pnl/trading_volume)*100:.2f}%"
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

def evaluate_withdrawal_request(agent, last_deposit, trading_volume, num_trades, current_pnl):
    """Get AI evaluation for a withdrawal request"""
    print("\nAnalyzing trading activity...")
    print("-" * 50)
    
    analysis = agent.analyze_activity(
        last_deposit, trading_volume, num_trades, current_pnl
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
    # Initialize the AI agent with historical fraud data
    agent = FraudDetectionAI('/Users/waqasyounas/Downloads/data.csv')
    
    print("AI Fraud Detection System Initialized")
    print("\nOptions:")
    print("1. Analyze withdrawal request")
    print("2. Add new pattern to database")
    print("3. Export fraud patterns report")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ")
            
            if choice == "1":
                deposit = float(input("Enter last deposit amount: "))
                trading_volume = float(input("Enter trading volume: "))
                num_trades = int(input("Enter number of trades: "))
                current_pnl = float(input("Enter current P&L: "))
                
                analysis = agent.analyze_activity(
                    deposit, trading_volume, num_trades, current_pnl
                )
                agent.update_pattern_stats(analysis)
                
                withdrawal_allowed = evaluate_withdrawal_request(
                    agent, deposit, trading_volume, num_trades, current_pnl
                )
                
                print(f"\nWithdrawal {'Allowed' if withdrawal_allowed else 'Flagged for Review'}")
                
            elif choice == "2":
                deposit = float(input("Enter deposit amount: "))
                trading_volume = float(input("Enter trading volume: "))
                num_trades = int(input("Enter number of trades: "))
                pnl = float(input("Enter P&L: "))
                is_fraudulent = input("Is this a fraudulent pattern? (y/n): ").lower() == 'y'
                
                agent.add_new_pattern(
                    deposit, trading_volume, num_trades, pnl, is_fraudulent
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

    print("\nThank you for using the Fraud Detection System")
