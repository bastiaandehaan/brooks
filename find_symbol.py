# diagnostics/find_symbol.py
"""
Debug script: Vind de correcte symbol naam in jouw MT5 terminal
"""

import MetaTrader5 as mt5
from utils.mt5_client import Mt5Client, Mt5ConnectionParams

def find_us500_symbol():
    """Find US500 symbol variants in your broker"""
    
    client = Mt5Client(mt5, Mt5ConnectionParams())
    
    if not client.initialize():
        print("❌ Failed to initialize MT5")
        return
    
    try:
        print("🔍 Searching for US500 variants...\n")
        
        # Search patterns
        patterns = [
            "US500",
            "SPX",
            "SP500",
            "S&P",
            "USA500",
            "USTEC",
        ]
        
        found_symbols = []
        
        for pattern in patterns:
            symbols = client.symbols_search(pattern)
            if symbols:
                print(f"✅ Found {len(symbols)} symbols matching '{pattern}':")
                for sym in symbols:
                    print(f"   - {sym}")
                    found_symbols.extend(symbols)
        
        # Also try getting all symbols and filter
        print("\n🔍 Checking all available symbols for '500' or 'SPX'...")
        all_symbols = client.symbols_search("")
        
        relevant = [s for s in all_symbols if '500' in s.upper() or 'SPX' in s.upper() or 'S&P' in s.upper()]
        
        if relevant:
            print(f"\n✅ Found {len(relevant)} relevant symbols:")
            for sym in relevant[:20]:  # Show first 20
                print(f"   - {sym}")
        
        if not found_symbols and not relevant:
            print("\n❌ No US500 variants found.")
            print("\n💡 All available symbols (first 50):")
            for sym in all_symbols[:50]:
                print(f"   - {sym}")
        
        print("\n" + "="*60)
        print("INSTRUCTIES:")
        print("="*60)
        print("1. Zoek bovenstaande lijst voor jouw S&P 500 symbol")
        print("2. Kopieer de exacte naam (case-sensitive!)")
        print("3. Update quick_audit.py regel met:")
        print('   symbol="[JOUW_SYMBOL_NAAM]"')
        print("\nVoorbeeld symbolen per broker:")
        print("  - FTMO: vaak 'US500' of 'US500.cash'")
        print("  - IC Markets: vaak 'US500' of 'USIDXUSD'")
        print("  - Pepperstone: vaak 'US500' of 'US500.f'")
        
    finally:
        client.shutdown()


if __name__ == "__main__":
    find_us500_symbol()
