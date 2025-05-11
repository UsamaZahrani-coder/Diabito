declare module '@react-native-community/datetimepicker' {
  import { ComponentType } from 'react';

  export interface DateTimePickerEvent {
    type: string;
    nativeEvent: {
      timestamp?: number;
      utcOffset?: number;
    };
  }

  export interface DateTimePickerProps {
    value: Date;
    mode?: 'date' | 'time' | 'datetime' | 'countdown';
    display?: 'default' | 'spinner' | 'calendar' | 'clock';
    onChange?: (event: DateTimePickerEvent, date?: Date) => void;
    maximumDate?: Date;
    minimumDate?: Date;
    timeZoneOffsetInMinutes?: number;
    textColor?: string;
    accentColor?: string;
    neutralButton?: boolean;
    negativeButton?: boolean;
    negativeButtonLabel?: string;
    positiveButtonLabel?: string;
    locale?: string;
    is24Hour?: boolean;
    minuteInterval?: 1 | 2 | 3 | 4 | 5 | 6 | 10 | 12 | 15 | 20 | 30;
    style?: any;
    disabled?: boolean;
  }

  const DateTimePicker: ComponentType<DateTimePickerProps>;
  export default DateTimePicker;
}